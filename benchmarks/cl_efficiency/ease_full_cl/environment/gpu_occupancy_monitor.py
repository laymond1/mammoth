import csv,datetime,hashlib,json,re,subprocess,sys,time
from pathlib import Path
run=Path(sys.argv[1]);output=Path(sys.argv[2]);uuid=sys.argv[3]
owner=None;samples=[];errors=[]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(final=False):
 foreign=[{'timestamp_utc':s['timestamp_utc'],'pids':[p['pid'] for p in s['processes'] if p['pid']!=owner]} for s in samples if owner is not None and any(p['pid']!=owner for p in s['processes'])]
 completion=json.loads((run/'completion.json').read_text()) if (run/'completion.json').is_file() else None
 record={'schema_version':1,'status':('pass' if owner and samples and not errors and not foreign and completion and completion['status']=='completed' else 'fail') if final else 'running','physical_gpu':1,'gpu_uuid':uuid,'worker_pid':owner,'run_dir':str(run.resolve()),'spec_sha256':digest(run/'spec.json'),'result_sha256':digest(run/'result.json') if final and (run/'result.json').is_file() else None,'monitor_code_sha256':digest(Path(__file__)),'interval_s':5,'measurement_scope':'Compute-process sampling every five seconds, plus empty-GPU preflight; this is sampled evidence, not a device lock or continuous reservation. Observer does no GPU compute.','samples':samples,'foreign_events':foreign,'errors':errors,'completion':completion}
 output.parent.mkdir(parents=True,exist_ok=True)
 temporary=output.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(output)
while True:
 try:
  matches=re.findall(r'Main process PID:\s*(\d+)',(run/'console.log').read_text(errors='replace'))
  if matches:owner=int(matches[-1])
  command=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=5)
  processes=[]
  for row in csv.reader(command.stdout.strip().splitlines()):
   if len(row)==3 and row[1].strip()==uuid:
    processes.append({'pid':int(row[0].strip()),'used_memory_mib':float(row[2].strip())})
  samples.append({'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'processes':processes})
 except Exception as error:errors.append({'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'type':type(error).__name__})
 final=(run/'completion.json').is_file();save(final)
 if final:break
 time.sleep(5)
print(json.dumps({'status':json.loads(output.read_text())['status'],'samples':len(samples),'worker_pid':owner,'receipt':str(output)}))
