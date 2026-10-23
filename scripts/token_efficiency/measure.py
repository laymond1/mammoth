"""Latency, memory, kernel and power measurement for CUDA GPUs, Jetson boards and CPU-only boards.

Power sources, chosen by what the device offers:
  NVML (desktop GPUs, GB10)  board energy counter and power, read through ctypes (no pynvml needed)
  tegrastats (Jetson)        VDD_IN, the whole board's input power, every 100 ms
  vcgencmd (Raspberry Pi 5)  sum of current x voltage over the PMIC rails (misses USB devices)
"""
import ctypes
import os
import platform
import re
import shutil
import statistics
import subprocess
import threading
import time

import torch


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def run_for(fn, x, device, min_iters, min_seconds, max_iters=100000):
    """Per-iteration wall times (seconds), each ending in a device synchronisation."""
    times, start = [], time.perf_counter()
    while len(times) < min_iters or time.perf_counter() - start < min_seconds:
        begin = time.perf_counter()
        fn(x)
        sync(device)
        times.append(time.perf_counter() - begin)
        if len(times) >= max_iters:
            break
    return times


def summary(times):
    ordered = sorted(times)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]
    return {'median_ms': statistics.median(times) * 1e3, 'mean_ms': statistics.fmean(times) * 1e3,
            'p10_ms': pick(0.1) * 1e3, 'p90_ms': pick(0.9) * 1e3, 'p99_ms': pick(0.99) * 1e3, 'iterations': len(times)}


def capture_graph(fn, x):
    """CUDA Graph of ``fn(x)``: replaying it removes the CPU's kernel-launch cost."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn(x)
    return graph


KERNEL_GROUPS = [
    ('attention', re.compile(r'flash|fmha|attention|efficient', re.I)),
    ('gemm', re.compile(r'gemm|cublas|cutlass|xmma|s16816|s1688|h16816|h1688|wgmma|matmul|gemv|splitk', re.I)),
    ('conv', re.compile(r'conv|implicit|winograd|cudnn', re.I)),
    ('sort-select', re.compile(r'sort|topk|radix|bitonic|select', re.I)),
    ('gather-scatter', re.compile(r'gather|scatter|index|cat|copy', re.I)),
    ('norm-softmax', re.compile(r'norm|softmax', re.I)),
]


def kernel_profile(fn, x):
    """Kernels of one call: count, GPU time per group, and the attention kernels that ran."""
    from torch.profiler import ProfilerActivity, profile
    fn(x)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(x)
        torch.cuda.synchronize()
    kernels = [event for event in prof.events() if event.device_type == torch.autograd.DeviceType.CUDA]
    groups = {}
    for event in kernels:
        group = next((name for name, pattern in KERNEL_GROUPS if pattern.search(event.name)), 'other-elementwise')
        entry = groups.setdefault(group, {'kernels': 0, 'gpu_us': 0.0})
        entry['kernels'] += 1
        entry['gpu_us'] += event.device_time
    attention = sorted({event.name[:80] for event in kernels if KERNEL_GROUPS[0][1].search(event.name)})
    return {'kernels': len(kernels), 'gpu_us': sum(event.device_time for event in kernels), 'groups': groups,
            'attention_kernels': attention}


def memory_reset(device):
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)


def memory_peak(device):
    out = {}
    if device.type == 'cuda':
        out['cuda_peak_allocated_mb'] = torch.cuda.max_memory_allocated(device) / 2 ** 20
        out['cuda_peak_reserved_mb'] = torch.cuda.max_memory_reserved(device) / 2 ** 20
    out.update(process_memory())
    return out


def process_memory():
    """Resident set size now and its peak since the process started (Linux)."""
    out = {}
    try:
        for line in open('/proc/self/status'):
            if line.startswith(('VmRSS:', 'VmHWM:')):
                key = 'rss_mb' if line.startswith('VmRSS') else 'rss_peak_mb'
                out[key] = int(line.split()[1]) / 1024
    except OSError:
        pass
    return out


# ------------------------------------------------------------------------------------------- power
class Nvml:
    """Minimal NVML through ctypes: board energy counter (mJ), power (mW), SM clock and temperature."""

    def __init__(self, index):
        self.lib = ctypes.CDLL('libnvidia-ml.so.1')
        if self.lib.nvmlInit_v2() != 0:
            raise OSError('nvmlInit failed')
        self.handle = ctypes.c_void_p()
        if self.lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(self.handle)) != 0:
            raise OSError(f'No NVML device {index}')

    def _read(self, function, kind, *args):
        value = kind()
        return value.value if getattr(self.lib, function)(self.handle, *args, ctypes.byref(value)) == 0 else None

    def energy_mj(self):
        return self._read('nvmlDeviceGetTotalEnergyConsumption', ctypes.c_ulonglong)

    def power_mw(self):
        return self._read('nvmlDeviceGetPowerUsage', ctypes.c_uint)

    def sm_clock_mhz(self):
        return self._read('nvmlDeviceGetClockInfo', ctypes.c_uint, ctypes.c_int(1))

    def memory_clock_mhz(self):
        return self._read('nvmlDeviceGetClockInfo', ctypes.c_uint, ctypes.c_int(2))

    def temperature_c(self):
        return self._read('nvmlDeviceGetTemperature', ctypes.c_uint, ctypes.c_int(0))


class Sampler:
    """Background power samples (time, watts); ``energy`` integrates them over an interval."""
    name = 'none'

    def __init__(self):
        self.samples, self._stop, self._thread = [], threading.Event(), None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self):
        raise NotImplementedError

    def energy(self, begin, end):
        """Trapezoidal integral of the samples between ``begin`` and ``end`` (perf_counter seconds)."""
        points = [(t, w) for t, w in self.samples if begin <= t <= end]
        if len(points) < 2:
            return None, len(points)
        joules = sum((t1 - t0) * (w0 + w1) / 2 for (t0, w0), (t1, w1) in zip(points, points[1:]))
        covered = points[-1][0] - points[0][0]
        return joules * (end - begin) / covered if covered > 0 else None, len(points)


class NvmlSampler(Sampler):
    name = 'nvml'

    def __init__(self, nvml, interval=0.05):
        super().__init__()
        self.nvml, self.interval = nvml, interval

    def _loop(self):
        while not self._stop.is_set():
            power = self.nvml.power_mw()
            if power is not None:
                self.samples.append((time.perf_counter(), power / 1000))
            time.sleep(self.interval)


class TegrastatsSampler(Sampler):
    name = 'tegrastats'
    RAIL = re.compile(r'VDD_IN (\d+)mW')

    def __init__(self, interval_ms=100):
        super().__init__()
        self.interval_ms, self.process = interval_ms, None

    def _loop(self):
        self.process = subprocess.Popen(['tegrastats', '--interval', str(self.interval_ms)], stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL, text=True)
        for line in self.process.stdout:
            match = self.RAIL.search(line)
            if match:
                self.samples.append((time.perf_counter(), int(match.group(1)) / 1000))
            if self._stop.is_set():
                break
        self.process.terminate()


class PmicSampler(Sampler):
    name = 'rpi5-pmic'
    LINE = re.compile(r'^\s*(\S+)_([AV]) (?:current|volt)\(\d+\)=([\d.]+)[AV]', re.M)

    def __init__(self, interval=0.05):
        super().__init__()
        self.interval = interval

    @classmethod
    def read_watts(cls):
        text = subprocess.run(['vcgencmd', 'pmic_read_adc'], capture_output=True, text=True, timeout=5).stdout
        current, volt = {}, {}
        for rail, kind, value in cls.LINE.findall(text):
            (current if kind == 'A' else volt)[rail] = float(value)
        return sum(current[rail] * volt[rail] for rail in current if rail in volt)

    def _loop(self):
        while not self._stop.is_set():
            self.samples.append((time.perf_counter(), self.read_watts()))
            time.sleep(self.interval)


def power_source(device):
    """(name, sampler factory, nvml or None) for this device, or ('none', None, None)."""
    if device.type == 'cuda':
        try:
            index = int(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0])
            nvml = Nvml(index)
            if nvml.power_mw() is not None:
                return 'nvml', (lambda: NvmlSampler(nvml)), nvml
        except (OSError, ValueError):
            pass
    if shutil.which('tegrastats'):
        return 'tegrastats', TegrastatsSampler, None
    if shutil.which('vcgencmd'):
        try:
            PmicSampler.read_watts()
            return 'rpi5-pmic', PmicSampler, None
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return 'none', None, None


def measure_energy(fn, x, device, seconds, images_per_call):
    """Run ``fn`` for at least ``seconds`` and return energy per image and mean power."""
    name, factory, nvml = power_source(device)
    if factory is None:
        return {'power_source': 'none'}
    sampler = factory()
    sampler.start()
    time.sleep(0.5)
    counter_begin = nvml.energy_mj() if nvml else None
    begin = time.perf_counter()
    calls = len(run_for(fn, x, device, min_iters=1, min_seconds=seconds))
    end = time.perf_counter()
    counter_end = nvml.energy_mj() if nvml else None
    time.sleep(0.3)
    sampler.stop()
    joules, samples = sampler.energy(begin, end)
    out = {'power_source': name, 'seconds': end - begin, 'calls': calls, 'samples': samples}
    if counter_begin is not None and counter_end is not None and counter_end > counter_begin:
        joules = (counter_end - counter_begin) / 1000
        out['energy_from'] = 'nvml-counter'
    if joules is not None:
        out.update(joules=joules, mean_w=joules / (end - begin), joules_per_image=joules / (calls * images_per_call))
    return out


def idle_power(device, seconds=5.0):
    name, factory, _ = power_source(device)
    if factory is None:
        return {'power_source': 'none'}
    sampler = factory()
    sampler.start()
    time.sleep(0.5)
    begin = time.perf_counter()
    time.sleep(seconds)
    end = time.perf_counter()
    sampler.stop()
    joules, samples = sampler.energy(begin, end)
    return {'power_source': name, 'idle_w': joules / (end - begin) if joules else None, 'samples': samples}


# ----------------------------------------------------------------------------------------- context
def _read(path):
    try:
        return open(path).read().strip()
    except (OSError, TypeError, ValueError):
        # Some Jetson sysfs nodes fail inside the decoder instead of raising OSError.
        return None


def _run(*command):
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def other_gpu_processes(all_gpus=False):
    """PIDs of other compute processes on this process's GPU, or on every GPU of the machine."""
    if not shutil.which('nvidia-smi'):
        return []
    select = () if all_gpus else ('-i', os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0])
    out = _run('nvidia-smi', *select, '--query-compute-apps=pid', '--format=csv,noheader') or ''
    return [int(pid) for pid in out.split() if pid.isdigit() and int(pid) != os.getpid()]


def _cpu_ticks():
    ticks = {}
    for entry in os.listdir('/proc'):
        if entry.isdigit():
            try:
                fields = open(f'/proc/{entry}/stat').read().rsplit(')', 1)[1].split()
                ticks[int(entry)] = int(fields[11]) + int(fields[12])  # utime + stime
            except (OSError, IndexError, ValueError):
                pass
    return ticks


def _family(pid):
    """This process, its ancestors and its descendants: the measurement itself."""
    family, parent = {pid}, pid
    while parent > 1:
        try:
            parent = int(open(f'/proc/{parent}/stat').read().rsplit(')', 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            break
        family.add(parent)
    changed = True
    while changed:
        changed = False
        for entry in os.listdir('/proc'):
            if entry.isdigit() and int(entry) not in family:
                try:
                    if int(open(f'/proc/{entry}/stat').read().rsplit(')', 1)[1].split()[1]) in family:
                        family.add(int(entry))
                        changed = True
                except (OSError, IndexError, ValueError):
                    pass
    return family


def busy_cpu_processes(window=1.0, threshold=0.5):
    """Other processes that used more than ``threshold`` of a core during ``window`` seconds."""
    hertz = os.sysconf('SC_CLK_TCK')
    before = _cpu_ticks()
    time.sleep(window)
    after = _cpu_ticks()
    family = _family(os.getpid())
    return sorted(pid for pid, ticks in after.items()
                  if pid not in family and (ticks - before.get(pid, ticks)) / hertz / window > threshold)


def jetson_gpu_load(seconds=2):
    """Mean GR3D (GPU) load in percent from tegrastats, or None without tegrastats."""
    if not shutil.which('tegrastats'):
        return None
    try:
        out = subprocess.run(['timeout', str(seconds + 1), 'tegrastats', '--interval', '500'], capture_output=True, text=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    loads = [int(value) for value in re.findall(r'GR3D_FREQ (\d+)%', out)]
    return sum(loads) / len(loads) if loads else None


def device_state(device):
    """Clocks, temperatures and throttling state at this moment."""
    state = {'cpu_governor': _read('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'),
             'cpu_mhz': (lambda v: int(v) / 1000 if v else None)(_read('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq'))}
    zones = {}
    for zone in sorted(os.listdir('/sys/class/thermal')) if os.path.isdir('/sys/class/thermal') else []:
        kind, temp = _read(f'/sys/class/thermal/{zone}/type'), _read(f'/sys/class/thermal/{zone}/temp')
        if kind and temp and temp.lstrip('-').isdigit():
            zones[kind] = int(temp) / 1000
    state['thermal_c'] = zones
    if device.type == 'cuda':
        state['other_gpu_processes'] = other_gpu_processes()
        state['other_gpu_processes_machine'] = other_gpu_processes(all_gpus=True)
        try:
            nvml = Nvml(int(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]))
            state.update(gpu_sm_mhz=nvml.sm_clock_mhz(), gpu_mem_mhz=nvml.memory_clock_mhz(),
                         gpu_temp_c=nvml.temperature_c(), gpu_power_w=(nvml.power_mw() or 0) / 1000)
        except (OSError, ValueError):
            pass
    if shutil.which('vcgencmd'):
        state['rpi_throttled'] = _run('vcgencmd', 'get_throttled')
    if shutil.which('nvpmodel'):
        state['nvpmodel'] = _run('nvpmodel', '-q')
        freq = _read('/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/cur_freq')
        state['jetson_gpu_mhz'] = int(freq) / 1e6 if freq and freq.isdigit() else None
    return state


def environment(device):
    info = {'hostname': platform.node(), 'machine': platform.machine(), 'python': platform.python_version(),
            'torch': torch.__version__, 'threads': torch.get_num_threads(), 'cpu_count': os.cpu_count()}
    model = _read('/proc/device-tree/model')
    if model:
        info['board'] = model.strip('\x00')
    cpu = [line.split(':', 1)[1].strip() for line in (_read('/proc/cpuinfo') or '').splitlines() if line.startswith('model name')]
    info['cpu'] = cpu[0] if cpu else platform.processor()
    if device.type == 'cuda':
        properties = torch.cuda.get_device_properties(device)
        info.update(gpu=properties.name, capability=f'{properties.major}.{properties.minor}', cuda=torch.version.cuda,
                    gpu_memory_mb=properties.total_memory / 2 ** 20)
    return info
