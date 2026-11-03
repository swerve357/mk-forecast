import sys, unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from forecast import per_ride_frames
from import_history import replay

class Integrity(unittest.TestCase):
    def test_gaps_and_closed_states_are_not_interpolated(self):
        ts=pd.to_datetime(['2026-10-30T13:00Z','2026-10-30T13:10Z','2026-10-30T13:15Z','2026-10-30T13:30Z'])
        df=pd.DataFrame(dict(ts=ts,ride_id=['a']*4,name=['A']*4,status=['OPERATING','CLOSED','OPERATING','OPERATING'],wait=[10,100,20,30]))
        f=per_ride_frames(df)
        self.assertTrue(pd.isna(f.loc[pd.Timestamp('2026-10-30T13:05Z'),'wait']))
        self.assertTrue(pd.isna(f.loc[ts[1],'wait']))
        self.assertEqual(f.loc[ts[0],'y30'],30)
    def test_replay_never_looks_forward_or_carries_indefinitely(self):
        data={'range':{'from':'2026-10-30','to':'2026-10-30'},'entities':[{'id':'a','name':'A','entityType':'ATTRACTION','history':[{'time':'2026-10-30T13:02:00Z','status':'OPERATING','queue':{'STANDBY':{'waitTime':10}}}]}]}
        rows=list(replay(data))
        self.assertEqual(len(rows),6)
        self.assertEqual(rows[0][0],'2026-10-30T13:05:00+00:00')
        self.assertEqual(rows[-1][0],'2026-10-30T13:30:00+00:00')

if __name__=='__main__': unittest.main()
