import sys,tempfile,unittest,subprocess,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from contract_payload_store import PayloadStore
class PayloadTests(unittest.TestCase):
 def test_cross_instance_retention_identity_and_bounds(self):
  with tempfile.TemporaryDirectory() as root:
   producer=PayloadStore(root,max_items=1,max_bytes=1024)
   consumer=PayloadStore(root,max_items=1,max_bytes=1024)
   identity={'token':'lease-a','revision':'scene-v3','kernel':'sha','sample_prefix':'pass-4'}
   handle,digest=producer.publish(identity,{'rays':[[1,2,3]],'complete':False})
   self.assertEqual(consumer.read(handle,identity,digest=digest)['rays'],[[1,2,3]])
   with self.assertRaises(ValueError):consumer.read(handle,{**identity,'token':'other'})
   with self.assertRaises(ValueError):consumer.read(handle,identity,digest='wrong')
   with self.assertRaises(BufferError):producer.publish(identity,{})
   script="import sys,json;sys.path.insert(0,sys.argv[1]);from contract_payload_store import PayloadStore;print(json.dumps(PayloadStore(sys.argv[2],max_bytes=1024).read(int(sys.argv[3]),json.loads(sys.argv[4]),digest=sys.argv[5])))"
   child=subprocess.run([sys.executable,'-c',script,str(Path(__file__).resolve().parents[1]/'tools'),root,str(handle),json.dumps(identity),digest],capture_output=True,text=True,check=True)
   self.assertEqual(json.loads(child.stdout)['rays'],[[1,2,3]])
   self.assertTrue(producer._path(handle).exists())
   producer.release(handle)
   with self.assertRaises(FileNotFoundError):consumer.read(handle,identity)
 def test_no_paths_or_nonfinite_payload(self):
  with tempfile.TemporaryDirectory() as root:
   store=PayloadStore(root,max_bytes=32)
   with self.assertRaises(ValueError):store.read('../config',{})
   with self.assertRaises(ValueError):store.publish({'a':1},float('nan'))
   with self.assertRaises(ValueError):store.publish({'a':1},'x'*80)
if __name__=='__main__':unittest.main()

