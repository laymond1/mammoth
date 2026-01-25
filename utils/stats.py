try:
    from resource import getrusage, RUSAGE_CHILDREN, RUSAGE_SELF

    def get_memory_mb():
        """
        Get the memory usage of the current process and its children.

        Returns:
            dict: A dictionary containing the memory usage of the current process and its children.

            The dictionary has the following keys:
                - self: The memory usage of the current process.
                - children: The memory usage of the children of the current process.
                - total: The total memory usage of the current process and its children.
        """
        res = {
            "self": getrusage(RUSAGE_SELF).ru_maxrss / 1024,
            "children": getrusage(RUSAGE_CHILDREN).ru_maxrss / 1024,
            "total": getrusage(RUSAGE_SELF).ru_maxrss / 1024 + getrusage(RUSAGE_CHILDREN).ru_maxrss / 1024
        }
        return res
except BaseException:
    get_memory_mb = None

try:
    import subprocess
    import re
    import threading
    import time
    import os
    from typing import Optional

    # -------------------------------
    # Jetson Power Monitor (기존)
    # -------------------------------
    class JetsonPowerMonitor:
        def __init__(self, interval_ms: int = 200):
            self.interval_ms = max(50, int(interval_ms))
            self.proc: Optional[subprocess.Popen] = None
            self._last_line: Optional[str] = None
            self._stop_event = threading.Event()
            self._lock = threading.Lock()
            self._reader_thread: Optional[threading.Thread] = None

            self.TOTAL_KEYS = [
                re.compile(r"(?:POM_5V_IN|VDD_IN)\s*[:=]?\s*(?P<mw>\d+)\s*mW"),
                re.compile(r"(?:POM_5V_IN|VDD_IN)\s*[:=]?\s*(?P<mw>\d+)(?=/)"),
            ]
            self.CPU_GPU_RE = re.compile(r"VDD_CPU_GPU_CV\s*(\d+)mW/(\d+)mW")
            self.SOC_RE = re.compile(r"VDD_SOC\s*(\d+)mW/(\d+)mW")

        def start(self):
            if self.proc is not None:
                return
            self.proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )

            def _reader():
                for line in self.proc.stdout:
                    if self._stop_event.is_set():
                        break
                    with self._lock:
                        self._last_line = line.strip()

            self._reader_thread = threading.Thread(target=_reader, daemon=True)
            self._reader_thread.start()

        def read_power_w(self) -> Optional[float]:
            with self._lock:
                line = self._last_line
            if not line:
                return None
            for rk in self.TOTAL_KEYS:
                m = rk.search(line)
                if m:
                    return float(m.group("mw")) / 1000.0
            return None

        def stop(self):
            self._stop_event.set()
            if self.proc:
                self.proc.terminate()
            if self._reader_thread:
                self._reader_thread.join(timeout=1)

    # -------------------------------
    # NVIDIA Power Monitor (RTX 4090)
    # -------------------------------
    class NvidiaPowerMonitor:
        def __init__(self, interval_ms: int = 200, gpu_index: int = 0):
            self.interval_ms = max(50, int(interval_ms))
            self.gpu_index = gpu_index
            self._last_power_w: Optional[float] = None
            self._stop_event = threading.Event()
            self._lock = threading.Lock()
            self._thread: Optional[threading.Thread] = None

        def start(self):
            if self._thread is not None:
                return

            def _worker():
                while not self._stop_event.is_set():
                    try:
                        res = subprocess.run(
                            [
                                "nvidia-smi",
                                f"--id={self.gpu_index}",
                                "--query-gpu=power.draw",
                                "--format=csv,noheader,nounits",
                            ],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL,
                            text=True,
                            timeout=1.0,
                        )
                        with self._lock:
                            self._last_power_w = float(res.stdout.strip())
                    except Exception:
                        pass
                    time.sleep(self.interval_ms / 1000)

            self._thread = threading.Thread(target=_worker, daemon=True)
            self._thread.start()

        def read_power_w(self) -> Optional[float]:
            with self._lock:
                return self._last_power_w

        def stop(self):
            self._stop_event.set()
            if self._thread:
                self._thread.join(timeout=1)

    # -------------------------------
    # Platform Detection
    # -------------------------------
    def _is_jetson():
        return os.path.exists("/usr/bin/tegrastats") or os.path.exists("/usr/sbin/tegrastats")

    # -------------------------------
    # Unified API
    # -------------------------------
    def get_power_w():
        try:
            monitor = JetsonPowerMonitor() if _is_jetson() else NvidiaPowerMonitor()
            monitor.start()
            time.sleep(0.4)
            power = monitor.read_power_w()
            monitor.stop()
            return power
        except Exception:
            return None

except BaseException:
    get_power_w = None
    JetsonPowerMonitor = None
    NvidiaPowerMonitor = None

try:
    import torch

    if torch.cuda.is_available():
        def get_alloc_memory_by_torch() -> list[int]:
            """
            Returns GPU memory allocated by the current PyTorch process.
            Values are in Bytes.
            """
            allocated = []
            for i in range(torch.cuda.device_count()):
                # _ = torch.tensor([1], device=f'cuda:{i}')  # force context init
                allocated.append(torch.cuda.max_memory_allocated(i))

            return allocated

        def get_memory_gpu_mb():
            """
            Get the memory usage of all GPUs in MB.
            """

            return [d / 1024 / 1024 for d in get_alloc_memory_by_torch()]
    else:
        get_memory_gpu_mb = None
except BaseException:
    get_memory_gpu_mb = None

try:
    from utils.conf import is_pynvml_available

    if is_pynvml_available():
        def get_memory_gpu_mb_pynvml_all() -> list[float]:
            """
            Get the GPU memory usage (in MB) for the current process on all GPUs as a list.

            Returns:
                List[float]: GPU memory usage per GPU (in MB) for the current process.
            """
            import os
            import torch
            from utils.conf import _get_gpu_memory_pynvml_all_processes

            current_pid = os.getpid()
            device_count = torch.cuda.device_count()
            results = []

            for device_id in range(device_count):
                handle = getattr(_get_gpu_memory_pynvml_all_processes, f'handle_{device_id}')
                procs = torch.cuda.pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                
                # Filter only current process and compute memory usage in MB
                mem_usage = [
                    proc.usedGpuMemory / 1024**2
                    for proc in procs
                    if proc.pid == current_pid
                ]

                # free, total = torch.cuda.mem_get_info(device_id) # force context init
                # mem_usage = [total / 1024**2 - free / 1024**2] if not mem_usage else mem_usage

                results.append(mem_usage[0] if mem_usage else 0.0)

            return results
    else:
        def get_memory_gpu_mb_jetson():
            """
            Get the GPU memory usage (in MB) for the current process on all GPUs as a list.
            This is a fallback for Jetson devices where pynvml is not available.
            """
            # DOTO: its function is not the exact solution, but it works for Jetson devices.
            import torch

            device_count = torch.cuda.device_count()
            results = []

            for device_id in range(device_count):
                free, total = torch.cuda.mem_get_info(device_id) # force context init
                used = total - free
                results.append(used / 1024**2)

            return results
        
        get_memory_gpu_mb_pynvml_all = get_memory_gpu_mb_jetson

except BaseException:
    get_memory_gpu_mb_pynvml_all = None

from utils.loggers import Logger


class track_system_stats:
    """
    A context manager that tracks the memory usage of the system.
    Tracks both CPU and GPU memory usage if available.

    Usage:

    .. code-block:: python

        with track_system_stats() as t:
            for i in range(100):
                ... # Do something
                t()

            cpu_res, gpu_res = t.cpu_res, t.gpu_res

    Args:
        logger (Logger): external logger.
        disabled (bool): If True, the context manager will not track the memory usage.
    """

    def get_stats(self):
        """
        Get the memory usage and power consumption of the system.

        Returns:
            tuple: (cpu_res, gpu_res, gpu_res_pynvml, power_res) where:
                - cpu_res: CPU memory usage in MB
                - gpu_res: GPU memory usage in MB  
                - gpu_res_pynvml: GPU memory usage via pynvml in MB
                - power_res: Power consumption in Watts
        """
        cpu_res = None
        if get_memory_mb is not None:
            cpu_res = get_memory_mb()['total']

        gpu_res = None
        if get_memory_gpu_mb is not None:
            gpu_res = get_memory_gpu_mb()
            
        gpu_res_pynvml = None
        if get_memory_gpu_mb_pynvml_all is not None:
            gpu_res_pynvml = get_memory_gpu_mb_pynvml_all()

        power_res = None
        if self._power_monitor is not None:
            power_res = self._power_monitor.read_power_w()
            # If monitor returns None, try fallback
            if power_res is None:
                power_res = get_power_w()
        elif get_power_w is not None:
            power_res = get_power_w()

        return cpu_res, gpu_res, gpu_res_pynvml, power_res

    def __init__(self, logger: Logger = None, disabled=False):
        self.logger = logger
        self.disabled = disabled
        self._it = 0
        self._power_monitor = None

    def __enter__(self):
        if self.disabled:
            return self
        
        # Initialize power monitoring if available
        if get_power_w is not None:
            try:
                self._power_monitor = JetsonPowerMonitor() if _is_jetson() else NvidiaPowerMonitor()
                self._power_monitor.start()
            except Exception:
                self._power_monitor = None
        
        self.initial_cpu_res, self.initial_gpu_res, self.initial_gpu_res_pynvml, self.initial_power_res = self.get_stats()
        
        if self.initial_cpu_res is None and self.initial_gpu_res is None and self.initial_power_res is None:
            self.disabled = True
        else:
            if self.initial_gpu_res is not None:
                self.initial_gpu_res = {g: g_res for g, g_res in enumerate(self.initial_gpu_res)}
            if self.initial_gpu_res_pynvml is not None:
                self.initial_gpu_res_pynvml = {g: g_res for g, g_res in enumerate(self.initial_gpu_res_pynvml)}
            
            self.avg_gpu_res = self.initial_gpu_res
            self.avg_gpu_res_pynvml = self.initial_gpu_res_pynvml
            self.avg_cpu_res = self.initial_cpu_res
            self.avg_power_res = self.initial_power_res

            self.max_cpu_res = self.initial_cpu_res
            self.max_gpu_res = self.initial_gpu_res
            self.max_gpu_res_pynvml = self.initial_gpu_res_pynvml
            self.max_power_res = self.initial_power_res

            # Initialize energy tracking
            self._energy_j = 0.0
            self._last_power_time = time.time()

            if self.logger is not None:
                self.logger.log_system_stats(self.initial_cpu_res, self.initial_gpu_res, self.initial_gpu_res_pynvml, self.initial_power_res)

        return self

    def __call__(self):
        if self.disabled:
            return

        cpu_res, gpu_res, gpu_res_pynvml, power_res = self.get_stats()
        self.update_stats(cpu_res, gpu_res, gpu_res_pynvml, power_res)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.disabled:
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()  # this allows to raise errors triggered previously by the GPU

        cpu_res, gpu_res, gpu_res_pynvml, power_res = self.get_stats()
        self.update_stats(cpu_res, gpu_res, gpu_res_pynvml, power_res)
        
        # Stop power monitoring
        if self._power_monitor is not None:
            self._power_monitor.stop()
            self._power_monitor = None

    def update_stats(self, cpu_res, gpu_res, gpu_res_pynvml, power_res):
        """
        Update the memory usage and power consumption statistics.

        Args:
            cpu_res (float): The memory usage of the CPU.
            gpu_res (list): The memory usage of the GPUs.
            gpu_res_pynvml (list): The memory usage of the GPUs via pynvml.
            power_res (float): The power consumption in Watts.
        """
        if self.disabled:
            return

        self._it += 1
        alpha = 1 / self._it
        
        if self.initial_cpu_res is not None:
            self.avg_cpu_res = self.avg_cpu_res + alpha * (cpu_res - self.avg_cpu_res)
            self.max_cpu_res = max(self.max_cpu_res, cpu_res)

        if self.initial_gpu_res is not None:
            self.avg_gpu_res = {g: (g_res + alpha * (g_res - self.avg_gpu_res[g])) for g, g_res in enumerate(gpu_res)}
            self.max_gpu_res = {g: max(self.max_gpu_res[g], g_res) for g, g_res in enumerate(gpu_res)}
            gpu_res = {g: g_res for g, g_res in enumerate(gpu_res)}

        if self.initial_gpu_res_pynvml is not None:
            self.avg_gpu_res_pynvml = {g: (g_res + alpha * (g_res - self.avg_gpu_res_pynvml[g])) for g, g_res in enumerate(gpu_res_pynvml)}
            self.max_gpu_res_pynvml = {g: max(self.max_gpu_res_pynvml[g], g_res) for g, g_res in enumerate(gpu_res_pynvml)}
            gpu_res_pynvml = {g: g_res for g, g_res in enumerate(gpu_res_pynvml)}

        if self.initial_power_res is not None and power_res is not None:
            self.avg_power_res = self.avg_power_res + alpha * (power_res - self.avg_power_res)
            self.max_power_res = max(self.max_power_res, power_res)
            
            # Calculate energy consumption (Joules)
            current_time = time.time()
            if hasattr(self, '_last_power_time'):
                time_delta = current_time - self._last_power_time
                self._energy_j += power_res * time_delta
            self._last_power_time = current_time

        if self.logger is not None:
            self.logger.log_system_stats(cpu_res, gpu_res, gpu_res_pynvml, power_res)

    def print_stats(self):
        """
        Print the memory usage and power consumption statistics.
        """

        cpu_res, gpu_res, gpu_res_pynvml, power_res = self.get_stats()

        # Print initial, average, final, and max memory usage
        print("System stats:")
        if cpu_res is not None:
            print(f"\tInitial CPU memory usage: {self.initial_cpu_res:.2f} MB", flush=True)
            print(f"\tAverage CPU memory usage: {self.avg_cpu_res:.2f} MB", flush=True)
            print(f"\tFinal CPU memory usage: {cpu_res:.2f} MB", flush=True)
            print(f"\tMax CPU memory usage: {self.max_cpu_res:.2f} MB", flush=True)

        if gpu_res is not None:
            for gpu_id, g_res in enumerate(gpu_res):
                print(f"\tInitial GPU {gpu_id} memory usage: {self.initial_gpu_res[gpu_id]:.2f} MB", flush=True)
                print(f"\tAverage GPU {gpu_id} memory usage: {self.avg_gpu_res[gpu_id]:.2f} MB", flush=True)
                print(f"\tFinal GPU {gpu_id} memory usage: {g_res:.2f} MB", flush=True)
                print(f"\tMax GPU {gpu_id} memory usage: {self.max_gpu_res[gpu_id]:.2f} MB", flush=True)
                
        if gpu_res_pynvml is not None:
            for gpu_id, g_res in enumerate(gpu_res_pynvml):
                print(f"\tInitial GPU {gpu_id} memory usage (pynvml): {self.initial_gpu_res_pynvml[gpu_id]:.2f} MB", flush=True)
                print(f"\tAverage GPU {gpu_id} memory usage (pynvml): {self.avg_gpu_res_pynvml[gpu_id]:.2f} MB", flush=True)
                print(f"\tFinal GPU {gpu_id} memory usage (pynvml): {g_res:.2f} MB", flush=True)
                print(f"\tMax GPU {gpu_id} memory usage (pynvml): {self.max_gpu_res_pynvml[gpu_id]:.2f} MB", flush=True)
                
        if power_res is not None and self.initial_power_res is not None:
            print(f"\tInitial Power consumption: {self.initial_power_res:.2f} W", flush=True)
            print(f"\tAverage Power consumption: {self.avg_power_res:.2f} W", flush=True)
            print(f"\tFinal Power consumption: {power_res:.2f} W", flush=True)
            print(f"\tMax Power consumption: {self.max_power_res:.2f} W", flush=True)
            print(f"\tTotal Energy consumption: {self._energy_j:.2f} J", flush=True)
            print(f"\tTotal Energy consumption: {self._energy_j / 3600:.4f} Wh", flush=True)