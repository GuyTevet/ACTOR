import matplotlib.pyplot as plt
import yaml
import os

# collect data
dir_path = 'pretrained_models/humanact12'
dataset = os.path.basename(dir_path)
interp_types = ['nearest', 'linear', 'cubic']
metrics = None
eval_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.startswith('evaluation_') and f.endswith('.yaml')]
no_interp_file = [f for f in eval_files if f.endswith('all.yaml')][0]
no_interp_yaml = {}
with open(no_interp_file, "r") as stream:
    no_interp_yaml = yaml.safe_load(stream)['feats']
    metrics = list(no_interp_yaml.keys())
    for k in no_interp_yaml.keys():
        no_interp_yaml[k] = float(no_interp_yaml[k][0])
data_dict = {}
for type_name in interp_types:
    files = [f for f in eval_files if 'interp_{}'.format(type_name) in f]
    yamls = []
    for f in files:
        with open(f, "r") as stream:
            y = yaml.safe_load(stream)['feats']
            for k in y.keys():
                y[k] = float(y[k][0])
            yamls.append(y)
    ratios = [int(f.replace('.yaml', '').split('_')[-1]) for f in files]
    data_dict[type_name] = [(ratios[i], files[i], yamls[i]) for i in range(len(files))]
    data_dict[type_name].append((1, no_interp_file, no_interp_yaml))
    data_dict[type_name].sort(key=lambda e: e[0])

# visualize data
vis_dir = os.path.join(dir_path, 'vis')
if not os.path.isdir(vis_dir):
    os.makedirs(vis_dir)
for met in metrics:
    fig = plt.figure()
    fig.suptitle('Dataset: [{}], Metric: [{}]'.format(dataset, met))
    ax = fig.add_subplot()
    for type_name in interp_types:
        data_preserved = [1/d[0] for d in data_dict[type_name]]
        metric_value = [d[2][met] for d in data_dict[type_name]]
        ax.plot(data_preserved, metric_value, linestyle='--', marker='o', label=type_name)
        ax.set_xlabel('Data preserved after sampling')
        ax.set_ylabel('Metric value')
    ax.legend()
    out_path = os.path.join(vis_dir, dataset + '_' + met + '.png')
    fig.savefig(out_path)