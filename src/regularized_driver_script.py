import os 
import yaml

with open(os.getcwd() + "/parameters.yaml", "r") as yaml_file:
    all_parameters = yaml.safe_load(yaml_file)


model_architecture = all_parameters['model_architecture']
pruning_type = all_parameters['pruning_type']
prune_schedule = all_parameters['prune_schedule']
test_type = all_parameters['test_type']


if test_type == 'pruning_type':
    pruning_rate = [0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999]
elif test_type == 'pruning_schedule':
    pruning_rate = [0.99, 0.995, 0.999]


for i, prn_r in enumerate(pruning_rate):
    print(prn_r)
    if test_type == 'pruning_schedule':
        os.system('python main.py > results/pruning_schedule/'+model_architecture+'/'+prune_schedule+'_'+pruning_type+'_'+str(prn_r)+'.txt '+str(i)+' '+str(prn_r))
    elif test_type == 'pruning_type':
        os.system('python main.py > results/pruning_type/'+model_architecture+'/'+prune_schedule+'_'+pruning_type+'_'+str(prn_r)+'.txt '+str(i)+' '+str(prn_r))
 