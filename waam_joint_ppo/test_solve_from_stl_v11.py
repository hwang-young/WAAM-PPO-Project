"""Safety and input-independence smoke tests; no existing PASS bank needed."""
import csv
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from waam_joint_ppo import solve_from_stl_v11 as m


class TestFreshSTL(unittest.TestCase):
    def test_fastest_requires_newly_verified_pass(self):
        rows=[{'validation':{'passed':False,'makespan_s':1}},
              {'validation':{'passed':True,'makespan_s':55}},
              {'validation':{'passed':True,'makespan_s':70}}]
        self.assertIs(m.choose_fastest(rows),rows[1])

    def test_durations(self):
        self.assertEqual(m.format_duration(6704.7),'00:01:51:45')
        self.assertEqual(m.format_duration(None),'N/A')

    def test_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'trajectory.csv'
            with p.open('w',newline='') as f:
                w=csv.writer(f);w.writerow(('robot_id','time_s','x_mm','y_mm','z_mm','mode'))
                for r in (1,2,3):
                    modes=('D','W') if r!=3 else ('W','W')
                    w.writerow((r,0,0,0,0,modes[0]));w.writerow((r,10,0,0,0,modes[1]))
            self.assertEqual(m.simultaneous_deposition(p),10)

    def test_new_candidate_generation_ignores_cache(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); inp=root/'input';inp.mkdir()
            (inp/'target.stl').write_bytes(b'fake-stl')
            (inp/'config.yaml').write_text('fake: yaml')
            (root/'ppo_project'/'old_pass').mkdir(parents=True)
            (root/'ppo_project'/'old_pass'/'trajectory.csv').write_text('cached-but-must-not-be-read')
            def gen(command,cwd,log_dir,timeout):
                out=Path(command[command.index('--output')+1]);out.mkdir(parents=True)
                shutil.copy2(inp/'target.stl',out/'target.stl')
                shutil.copy2(inp/'config.yaml',out/'config.yaml')
                (out/'trajectory.csv').write_text('fresh-generated')
                return {'status':'FINISHED','wall_s':1.5,'exit_code':0,'stderr_tail':''}
            def validator(root,src,logs,timeout):
                self.assertNotIn('old_pass',str(src))
                self.assertTrue((Path(src)/'trajectory.csv').read_text()=='fresh-generated')
                return {'status':'FINISHED','report_status':'PASS','passed':True,
                        'makespan_s':123.4,'wall_s':2.0,'failure_reasons':[]}
            with patch.object(m,'run_command',side_effect=gen),patch.object(m,'official_validate',side_effect=validator),patch.object(m,'simultaneous_deposition',return_value=0.0):
                opts=SimpleNamespace(variants=1,generator_timeout_s=10,validator_timeout_s=10,
                                     search_probes=0)
                output=root/'outputs'/'model_a'
                result=m.solve_one(root,inp,'model_a',output,opts)
            self.assertEqual(result['status'],'PASS')
            self.assertAlmostEqual(result['makespan_s'],123.4)
            self.assertEqual(result['generation_wall_s'],1.5)
            self.assertEqual(result['validator_wall_s'],4.0)
            self.assertTrue((output/'selected_pass'/'trajectory.csv').is_file())

    def test_validator_fail_preserves_reasons(self):
        with tempfile.TemporaryDirectory() as d:
            logs=Path(d)/'logs';logs.mkdir()
            (logs/'stdout.txt').write_text('{"status":"FAIL","failure_reasons":["ARM_ENVELOPE"]}')
            record={'status':'PROCESS_ERROR','exit_code':1,'wall_s':2.0,'stderr_tail':''}
            with patch.object(m,'run_command',return_value=record):
                val=m.official_validate(Path(d),Path(d),logs,60)
            self.assertEqual(val['report_status'],'FAIL')
            self.assertEqual(val['failure_reasons'],['ARM_ENVELOPE'])
            self.assertFalse(val['passed'])

    def test_missing_input_report(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);bad=root/'bad';bad.mkdir()
            opts=SimpleNamespace(variants=1,generator_timeout_s=10,validator_timeout_s=10,search_probes=0)
            out=root/'result'
            r=m.solve_one(root,bad,'test',out,opts)
            self.assertEqual(r['status'],'ERROR')
            self.assertIsNone(r['makespan_s'])
            self.assertTrue((out/'result.json').is_file())


if __name__=='__main__':unittest.main()
