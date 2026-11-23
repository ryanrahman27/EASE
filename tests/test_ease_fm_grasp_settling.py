import numpy as np
from examples.ease_fm.grasp_settling import settling_sequence
from examples.ease_fm.arm_pick_place import targets


class Fixture:
    xy=np.array([.35,0]);target=np.array([.35,.06])
    def __init__(self,bilateral): self.bilateral=bilateral
    def observe(self): return {'position':np.array([.35,0,.24])}
    def contacts(self): return ({'left','right'} if self.bilateral else set()),[]


def test_baseline_sequence_unchanged():
    env=Fixture(True);events=[]
    actual=list(settling_sequence(env,'baseline',events))
    expected=targets(env.xy,env.target)
    assert len(actual)==len(expected) and not events
    for a,b in zip(actual,expected):
        assert a[0]==b[0] and np.array_equal(a[1],b[1]) and a[2:]==b[2:]


def test_bounded_contact_settling_and_fixed_control():
    for mode,bilateral,count in [('fixed',True,20),('contact',True,8),('contact',False,20)]:
        events=[];seq=list(settling_sequence(Fixture(bilateral),mode,events))
        assert len(seq)==9+2*count
        assert [e['steps'] for e in events]==[count,count]
        assert [e['before'] for e in events]==['lift','transport']


def test_contact_streak_resets():
    env=Fixture(True);events=[];seq=settling_sequence(env,'contact',events)
    for _ in range(3):next(seq)  # Original approach/descend/close.
    for i in range(20):
        item=next(seq)
        assert item[0]=='close'
        env.bilateral=i%7!=6
    assert next(seq)[0]=='lift'
    assert events[0]['steps']==20 and not events[0]['stable_contact']
