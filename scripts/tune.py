"""Equal-budget local Optuna trials. Development only; negative results are retained."""
import argparse
from src.training.runner import ROOT,parser,train

def main():
    import optuna
    p=argparse.ArgumentParser()
    p.add_argument('--algorithm',choices=('iql','vdn','qmix'),default='qmix')
    p.add_argument('--trials',type=int,default=10);p.add_argument('--steps',type=int,default=20000)
    p.add_argument('--num-agents',type=int,default=5);p.add_argument('--tiny',action='store_true')
    args=p.parse_args()
    if min(args.trials,args.steps,args.num_agents)<1:raise ValueError('Positive budgets required')
    output=ROOT/'outputs'/'tuning';output.mkdir(parents=True,exist_ok=True)
    def objective(trial):
        config=parser(args.algorithm).parse_args([])
        config.num_agents=args.num_agents;config.steps=args.steps;config.tiny=args.tiny
        config.learning_rate=trial.suggest_float('learning_rate',1e-5,1e-3,log=True)
        config.gamma=trial.suggest_categorical('gamma',[.95,.99,.995])
        if args.tiny:config.num_agents=1;config.stage='C0';config.horizon=30;config.eval_episodes=3
        _,metadata,results=train(config)
        trial.set_user_attr('run_id',metadata['run_id'])
        trial.set_user_attr('development_only',True)
        return sum(r['throughput'] for r in results)/len(results)
    study=optuna.create_study(study_name=f'{args.algorithm}_N{args.num_agents}_steps{args.steps}',storage=f'sqlite:///{(output/"study.sqlite3").as_posix()}',load_if_exists=True,direction='maximize',sampler=optuna.samplers.TPESampler(seed=11))
    study.optimize(objective,n_trials=args.trials,catch=(RuntimeError,ValueError))
    print(dict(best=study.best_params,value=study.best_value,trials=len(study.trials),scope='development tuning; no final test used'))

if __name__=='__main__':main()
