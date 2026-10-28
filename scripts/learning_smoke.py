"""Explicit tiny learning experiment; never presented as warehouse fleet validation."""
import json
from src.training.runner import parser,train,ROOT

def main():
    args=parser('qmix').parse_args([])
    args.num_agents=1;args.tiny=True;args.stage='C0';args.steps=400;args.horizon=30
    args.bc_episodes=10;args.bc_epochs=6;args.eval_episodes=5
    args.batch_size=4;args.burn_in=2;args.unroll=10;args.epsilon_steps=400
    _,metadata,results=train(args)
    directory=ROOT/'outputs'/'runs'/metadata['run_id']
    initial=json.loads((directory/'initial_development.json').read_text(encoding='utf-8'))
    before=sum(r['completion_rate'] for r in initial)/len(initial)
    after=sum(r['completion_rate'] for r in results)/len(results)
    report=dict(scope='tiny N=1 imitation plus real QMIX TD updates; not full-warehouse quality',
        initial_completion=before,final_completion=after,updates=metadata['updates'],
        passed=after>before and after>=.8 and metadata['updates']>=50,run_id=metadata['run_id'],
        source_sha256=metadata['provenance']['source_sha256'])
    target=ROOT/'outputs'/'evaluation'/'learning_smoke.json';target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2),encoding='utf-8');print(report)
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
