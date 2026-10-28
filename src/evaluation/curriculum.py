"""Executable curriculum order and training-only prerequisite case membership."""
STAGES=('C0','C0.5','C0.75','C1','C2','C3','C4','C5')
TRAINING_CASES={
    'C0':(), 'C0.5':(), 'C0.75':('COO01',),
    'C1':('COO01','COO02','COO04','COO05'), 'C2':('CHARGE_CYCLE','CHARGE_APPROACH','ROB03'),
    'C3':('HND06','HND_TRANSFER','ROB04'),
    'C4':('DEM02','DEM03','ROB01','ROB03'),
    'C5':('HND06','HND_TRANSFER','COO01','COO02','COO04','COO05',
          'DEM02','DEM03','ROB01','ROB03','ROB04','CHARGE_CYCLE','CHARGE_APPROACH','EVA05')}

def eligible_cases(stage,num_agents):
    return tuple(c for c in TRAINING_CASES[stage]
                 if num_agents>1 and (c!='COO01' or num_agents<=18))
