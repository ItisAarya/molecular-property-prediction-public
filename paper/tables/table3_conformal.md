
## Table 3: conformal coverage at a nominal 90%

| method               | tag                | dataset       | coverage | coverage_pos | coverage_neg | mean_set_size | mean_width | width_spread |
|----------------------|--------------------|---------------|----------|--------------|--------------|---------------|------------|--------------|
| marginal             | rf                 | tox21         | 90.7     |              |              | 0.977         |            |              |
| marginal             | rf                 | bbbp          | 90.1     |              |              | 1.056         |            |              |
| marginal             | rf                 | clintox       | 91.8     |              |              | 0.979         |            |              |
| marginal             | rf                 | esol          | 87.7     |              |              |               | 5.925      |              |
| marginal             | rf                 | lipophilicity | 88.1     |              |              |               | 3.112      |              |
| marginal             | gnn                | tox21         | 90.0     |              |              | 1.217         |            |              |
| marginal             | gnn                | bbbp          | 92.1     |              |              | 1.256         |            |              |
| marginal             | gnn                | clintox       | 92.2     |              |              | 1.097         |            |              |
| marginal             | gnn                | esol          | 93.7     |              |              |               | 3.952      |              |
| marginal             | gnn                | lipophilicity | 87.6     |              |              |               | 2.432      |              |
| marginal             | trf                | tox21         | 91.0     |              |              | 0.993         |            |              |
| marginal             | trf                | bbbp          | 85.1     |              |              | 1.315         |            |              |
| marginal             | trf                | clintox       | 92.6     |              |              | 0.943         |            |              |
| marginal             | trf                | esol          | 90.5     |              |              |               | 4.897      |              |
| marginal             | trf                | lipophilicity | 89.6     |              |              |               | 3.568      |              |
| marginal             | hybrid             | tox21         | 90.2     |              |              | 1.187         |            |              |
| marginal             | hybrid             | bbbp          | 90.9     |              |              | 1.23          |            |              |
| marginal             | hybrid             | clintox       | 90.9     |              |              | 0.994         |            |              |
| marginal             | hybrid             | esol          | 93.5     |              |              |               | 4.059      |              |
| marginal             | hybrid             | lipophilicity | 89.0     |              |              |               | 2.329      |              |
| marginal             | ens                | tox21         | 89.8     |              |              | 1.173         |            |              |
| marginal             | ens                | bbbp          | 90.7     |              |              | 1.153         |            |              |
| marginal             | ens                | clintox       | 92.6     |              |              | 0.943         |            |              |
| marginal             | ens                | esol          | 94.6     |              |              |               | 4.153      |              |
| marginal             | ens                | lipophilicity | 89.1     |              |              |               | 2.341      |              |
| absolute             | desc               | tox21         | 90.0     | 77.9         | 90.7         | 1.212         |            |              |
| absolute             | desc               | bbbp          | 90.5     | 93.1         | 79.3         | 1.057         |            |              |
| absolute             | desc               | clintox       | 90.8     | 79.2         | 82.8         | 1.139         |            |              |
| absolute             | desc               | esol          | 85.1     |              |              |               | 2.215      | 0.0          |
| absolute             | desc               | lipophilicity | 88.1     |              |              |               | 2.004      | 0.0          |
| absolute             | desc               | bace          | 85.7     | 89.6         | 81.5         | 1.201         |            |              |
| absolute             | desc               | sider         | 90.4     | 87.0         | 86.6         | 1.587         |            |              |
| absolute             | desc               | freesolv      | 90.2     |              |              |               | 2.918      | 0.0          |
| absolute             | fuse_gated         | tox21         | 90.3     | 72.4         | 91.3         | 1.172         |            |              |
| absolute             | fuse_gated         | bbbp          | 90.1     | 92.2         | 80.6         | 1.029         |            |              |
| absolute             | fuse_gated         | clintox       | 92.4     | 89.6         | 87.1         | 0.935         |            |              |
| absolute             | fuse_gated         | esol          | 82.3     |              |              |               | 2.345      | 0.0          |
| absolute             | fuse_gated         | lipophilicity | 87.4     |              |              |               | 2.076      | 0.0          |
| absolute             | fuse_gated         | bace          | 85.7     | 88.3         | 82.5         | 1.254         |            |              |
| absolute             | fuse_gated         | sider         | 90.4     | 86.1         | 85.8         | 1.583         |            |              |
| absolute             | fuse_gated         | freesolv      | 89.8     |              |              |               | 3.415      | 0.0          |
| absolute             | fuse_gated_nograph | tox21         | 89.9     | 71.1         | 91.1         | 1.162         |            |              |
| absolute             | fuse_gated_nograph | bbbp          | 90.7     | 91.9         | 84.5         | 1.018         |            |              |
| absolute             | fuse_gated_nograph | clintox       | 92.7     | 91.0         | 92.9         | 0.992         |            |              |
| absolute             | fuse_gated_nograph | esol          | 78.1     |              |              |               | 2.096      | 0.0          |
| absolute             | fuse_gated_nograph | lipophilicity | 88.2     |              |              |               | 2.025      | 0.0          |
| absolute             | fuse_gated_nograph | bace          | 84.7     | 88.1         | 81.3         | 1.162         |            |              |
| absolute             | fuse_gated_nograph | sider         | 89.8     | 85.5         | 88.3         | 1.616         |            |              |
| absolute             | fuse_gated_nograph | freesolv      | 89.2     |              |              |               | 2.781      | 0.0          |
| absolute             | fuse_proposed_gpu  | tox21         | 89.4     | 72.7         | 90.4         | 1.201         |            |              |
| absolute             | fuse_proposed_gpu  | bbbp          | 89.6     | 92.5         | 77.0         | 0.999         |            |              |
| absolute             | fuse_proposed_gpu  | clintox       | 92.1     | 80.9         | 75.0         | 0.94          |            |              |
| absolute             | fuse_proposed_gpu  | esol          | 82.1     |              |              |               | 2.222      | 0.0          |
| absolute             | fuse_proposed_gpu  | lipophilicity | 88.9     |              |              |               | 2.074      | 0.0          |
| absolute             | fuse_proposed_gpu  | bace          | 86.1     | 89.9         | 82.3         | 1.196         |            |              |
| absolute             | fuse_proposed_gpu  | sider         | 90.3     | 89.1         | 83.9         | 1.6           |            |              |
| absolute             | fuse_proposed_gpu  | freesolv      | 88.9     |              |              |               | 2.807      | 0.0          |
| absolute             | fuse_concat_gpu    | tox21         | 89.8     | 71.3         | 90.9         | 1.193         |            |              |
| absolute             | fuse_concat_gpu    | bbbp          | 91.0     | 89.9         | 94.8         | 1.128         |            |              |
| absolute             | fuse_concat_gpu    | clintox       | 93.5     | 86.3         | 89.6         | 0.955         |            |              |
| absolute             | fuse_concat_gpu    | esol          | 84.2     |              |              |               | 2.408      | 0.0          |
| absolute             | fuse_concat_gpu    | lipophilicity | 88.8     |              |              |               | 2.046      | 0.0          |
| absolute             | fuse_concat_gpu    | bace          | 81.6     | 85.4         | 77.1         | 1.13          |            |              |
| absolute             | fuse_concat_gpu    | sider         | 90.8     | 86.0         | 92.0         | 1.718         |            |              |
| absolute             | fuse_concat_gpu    | freesolv      | 90.2     |              |              |               | 3.26       | 0.0          |
| absolute             | fuse_xattn_gpu     | tox21         | 90.1     | 76.1         | 90.9         | 1.238         |            |              |
| absolute             | fuse_xattn_gpu     | bbbp          | 90.4     | 94.0         | 76.8         | 1.023         |            |              |
| absolute             | fuse_xattn_gpu     | clintox       | 92.2     | 82.6         | 75.4         | 0.936         |            |              |
| absolute             | fuse_xattn_gpu     | esol          | 85.1     |              |              |               | 2.333      | 0.0          |
| absolute             | fuse_xattn_gpu     | lipophilicity | 88.6     |              |              |               | 2.079      | 0.0          |
| absolute             | fuse_xattn_gpu     | bace          | 83.9     | 88.1         | 79.5         | 1.153         |            |              |
| absolute             | fuse_xattn_gpu     | sider         | 88.6     | 80.8         | 96.6         | 1.71          |            |              |
| absolute             | fuse_xattn_gpu     | freesolv      | 90.5     |              |              |               | 2.966      | 0.0          |
| absolute             | fuse_bilinear_gpu  | tox21         | 90.4     | 77.7         | 91.0         | 1.234         |            |              |
| absolute             | fuse_bilinear_gpu  | bbbp          | 91.1     | 93.3         | 80.7         | 1.01          |            |              |
| absolute             | fuse_bilinear_gpu  | clintox       | 90.1     | 87.1         | 80.1         | 0.95          |            |              |
| absolute             | fuse_bilinear_gpu  | esol          | 81.9     |              |              |               | 2.279      | 0.0          |
| absolute             | fuse_bilinear_gpu  | lipophilicity | 88.3     |              |              |               | 2.044      | 0.0          |
| absolute             | fuse_bilinear_gpu  | bace          | 84.2     | 86.2         | 82.9         | 1.18          |            |              |
| absolute             | fuse_bilinear_gpu  | sider         | 90.8     | 87.5         | 85.4         | 1.591         |            |              |
| absolute             | fuse_bilinear_gpu  | freesolv      | 88.6     |              |              |               | 3.137      | 0.0          |
| absolute             | gin_ref_gpu        | tox21         | 89.8     | 76.8         | 90.6         | 1.263         |            |              |
| absolute             | gin_ref_gpu        | bbbp          | 90.3     | 92.7         | 79.4         | 1.13          |            |              |
| absolute             | gin_ref_gpu        | clintox       | 90.2     | 82.6         | 82.7         | 1.12          |            |              |
| absolute             | gin_ref_gpu        | esol          | 85.8     |              |              |               | 3.086      | 0.0          |
| absolute             | gin_ref_gpu        | lipophilicity | 88.2     |              |              |               | 2.248      | 0.0          |
| absolute             | gin_ref_gpu        | bace          | 85.1     | 83.3         | 84.8         | 1.274         |            |              |
| absolute             | gin_ref_gpu        | sider         | 90.4     | 90.1         | 85.6         | 1.637         |            |              |
| absolute             | gin_ref_gpu        | freesolv      | 84.0     |              |              |               | 3.453      | 0.0          |
| absolute             | rf                 | tox21         | 90.7     | 9.8          | 96.6         | 0.977         |            |              |
| absolute             | rf                 | bbbp          | 90.1     | 98.2         | 58.8         | 1.056         |            |              |
| absolute             | rf                 | clintox       | 91.8     | 50.5         | 51.3         | 0.979         |            |              |
| absolute             | rf                 | esol          | 87.7     |              |              |               | 5.925      | 0.0          |
| absolute             | rf                 | lipophilicity | 88.1     |              |              |               | 3.112      | 0.0          |
| absolute             | gnn                | tox21         | 90.0     | 72.3         | 91.0         | 1.217         |            |              |
| absolute             | gnn                | bbbp          | 92.1     | 92.8         | 88.5         | 1.256         |            |              |
| absolute             | gnn                | clintox       | 92.2     | 82.0         | 82.2         | 1.097         |            |              |
| absolute             | gnn                | esol          | 93.7     |              |              |               | 3.952      | 0.0          |
| absolute             | gnn                | lipophilicity | 87.6     |              |              |               | 2.432      | 0.0          |
| absolute             | trf                | tox21         | 91.0     | 9.7          | 96.8         | 0.993         |            |              |
| absolute             | trf                | bbbp          | 85.1     | 82.5         | 96.1         | 1.315         |            |              |
| absolute             | trf                | clintox       | 92.6     | 68.5         | 68.3         | 0.943         |            |              |
| absolute             | trf                | esol          | 90.5     |              |              |               | 4.897      | 0.0          |
| absolute             | trf                | lipophilicity | 89.6     |              |              |               | 3.568      | 0.0          |
| absolute             | hybrid             | tox21         | 90.2     | 74.9         | 91.3         | 1.187         |            |              |
| absolute             | hybrid             | bbbp          | 90.9     | 90.2         | 93.4         | 1.23          |            |              |
| absolute             | hybrid             | clintox       | 90.9     | 89.3         | 88.7         | 0.994         |            |              |
| absolute             | hybrid             | esol          | 93.5     |              |              |               | 4.059      | 0.0          |
| absolute             | hybrid             | lipophilicity | 89.0     |              |              |               | 2.329      | 0.0          |
| absolute             | ens                | tox21         | 89.8     | 72.4         | 91.0         | 1.173         |            |              |
| absolute             | ens                | bbbp          | 90.7     | 92.6         | 84.0         | 1.153         |            |              |
| absolute             | ens                | clintox       | 92.6     | 68.5         | 68.3         | 0.943         |            |              |
| absolute             | ens                | esol          | 94.6     |              |              |               | 4.153      | 0.0          |
| absolute             | ens                | lipophilicity | 89.1     |              |              |               | 2.341      | 0.0          |
| absolute_aps         | desc               | bbbp          | 100.0    | 100.0        | 100.0        | 2.0           |            |              |
| absolute_raps        | desc               | bbbp          | 100.0    | 100.0        | 100.0        | 2.0           |            |              |
| conditional          | desc               | tox21         | 90.0     | 90.6         | 90.0         | 1.411         |            |              |
| conditional          | desc               | bbbp          | 91.3     | 92.9         | 83.9         | 1.122         |            |              |
| conditional          | desc               | clintox       | 89.5     | 87.1         | 91.7         | 1.545         |            |              |
| conditional          | desc               | esol          | 85.1     |              |              |               | 2.215      | 0.0          |
| conditional          | desc               | lipophilicity | 88.1     |              |              |               | 2.004      | 0.0          |
| conditional          | desc               | bace          | 85.4     | 86.1         | 83.7         | 1.229         |            |              |
| conditional          | desc               | sider         | 91.2     | 92.7         | 89.7         | 1.725         |            |              |
| conditional          | desc               | freesolv      | 90.2     |              |              |               | 2.918      | 0.0          |
| conditional          | fuse_gated         | tox21         | 90.1     | 88.8         | 90.2         | 1.402         |            |              |
| conditional          | fuse_gated         | bbbp          | 91.6     | 92.0         | 89.2         | 1.058         |            |              |
| conditional          | fuse_gated         | clintox       | 92.2     | 94.9         | 95.3         | 1.301         |            |              |
| conditional          | fuse_gated         | esol          | 82.3     |              |              |               | 2.345      | 0.0          |
| conditional          | fuse_gated         | lipophilicity | 87.4     |              |              |               | 2.076      | 0.0          |
| conditional          | fuse_gated         | bace          | 89.5     | 92.2         | 85.9         | 1.334         |            |              |
| conditional          | fuse_gated         | sider         | 90.6     | 90.4         | 90.6         | 1.735         |            |              |
| conditional          | fuse_gated         | freesolv      | 89.8     |              |              |               | 3.415      | 0.0          |
| conditional          | fuse_gated_nograph | tox21         | 89.8     | 88.4         | 89.9         | 1.398         |            |              |
| conditional          | fuse_gated_nograph | bbbp          | 91.6     | 91.9         | 88.8         | 1.032         |            |              |
| conditional          | fuse_gated_nograph | clintox       | 92.8     | 95.0         | 94.5         | 1.297         |            |              |
| conditional          | fuse_gated_nograph | esol          | 78.1     |              |              |               | 2.096      | 0.0          |
| conditional          | fuse_gated_nograph | lipophilicity | 88.2     |              |              |               | 2.025      | 0.0          |
| conditional          | fuse_gated_nograph | bace          | 86.6     | 90.9         | 81.7         | 1.22          |            |              |
| conditional          | fuse_gated_nograph | sider         | 90.3     | 90.9         | 90.0         | 1.716         |            |              |
| conditional          | fuse_gated_nograph | freesolv      | 89.2     |              |              |               | 2.781      | 0.0          |
| conditional          | fuse_proposed_gpu  | tox21         | 89.7     | 90.1         | 89.7         | 1.421         |            |              |
| conditional          | fuse_proposed_gpu  | bbbp          | 90.8     | 92.2         | 84.2         | 1.019         |            |              |
| conditional          | fuse_proposed_gpu  | clintox       | 91.8     | 89.2         | 91.6         | 1.195         |            |              |
| conditional          | fuse_proposed_gpu  | esol          | 82.1     |              |              |               | 2.222      | 0.0          |
| conditional          | fuse_proposed_gpu  | lipophilicity | 88.9     |              |              |               | 2.074      | 0.0          |
| conditional          | fuse_proposed_gpu  | bace          | 85.8     | 90.1         | 81.0         | 1.176         |            |              |
| conditional          | fuse_proposed_gpu  | sider         | 91.0     | 91.4         | 91.3         | 1.746         |            |              |
| conditional          | fuse_proposed_gpu  | freesolv      | 88.9     |              |              |               | 2.807      | 0.0          |
| conditional          | fuse_concat_gpu    | tox21         | 89.8     | 88.9         | 89.9         | 1.421         |            |              |
| conditional          | fuse_concat_gpu    | bbbp          | 90.6     | 91.3         | 88.5         | 1.061         |            |              |
| conditional          | fuse_concat_gpu    | clintox       | 93.2     | 92.6         | 94.5         | 1.207         |            |              |
| conditional          | fuse_concat_gpu    | esol          | 84.2     |              |              |               | 2.408      | 0.0          |
| conditional          | fuse_concat_gpu    | lipophilicity | 88.8     |              |              |               | 2.046      | 0.0          |
| conditional          | fuse_concat_gpu    | bace          | 85.3     | 88.9         | 81.0         | 1.187         |            |              |
| conditional          | fuse_concat_gpu    | sider         | 92.2     | 91.7         | 93.1         | 1.803         |            |              |
| conditional          | fuse_concat_gpu    | freesolv      | 90.2     |              |              |               | 3.26       | 0.0          |
| conditional          | fuse_xattn_gpu     | tox21         | 90.1     | 89.6         | 90.2         | 1.416         |            |              |
| conditional          | fuse_xattn_gpu     | bbbp          | 90.9     | 93.3         | 80.8         | 1.031         |            |              |
| conditional          | fuse_xattn_gpu     | clintox       | 90.9     | 92.4         | 92.9         | 1.234         |            |              |
| conditional          | fuse_xattn_gpu     | esol          | 85.1     |              |              |               | 2.333      | 0.0          |
| conditional          | fuse_xattn_gpu     | lipophilicity | 88.6     |              |              |               | 2.079      | 0.0          |
| conditional          | fuse_xattn_gpu     | bace          | 85.0     | 86.9         | 81.9         | 1.183         |            |              |
| conditional          | fuse_xattn_gpu     | sider         | 90.6     | 88.9         | 94.5         | 1.787         |            |              |
| conditional          | fuse_xattn_gpu     | freesolv      | 90.5     |              |              |               | 2.966      | 0.0          |
| conditional          | fuse_bilinear_gpu  | tox21         | 90.4     | 88.5         | 90.5         | 1.4           |            |              |
| conditional          | fuse_bilinear_gpu  | bbbp          | 93.2     | 93.6         | 90.5         | 1.072         |            |              |
| conditional          | fuse_bilinear_gpu  | clintox       | 89.5     | 93.2         | 92.2         | 1.214         |            |              |
| conditional          | fuse_bilinear_gpu  | esol          | 81.9     |              |              |               | 2.279      | 0.0          |
| conditional          | fuse_bilinear_gpu  | lipophilicity | 88.3     |              |              |               | 2.044      | 0.0          |
| conditional          | fuse_bilinear_gpu  | bace          | 84.9     | 88.0         | 81.2         | 1.203         |            |              |
| conditional          | fuse_bilinear_gpu  | sider         | 90.9     | 90.6         | 91.4         | 1.744         |            |              |
| conditional          | fuse_bilinear_gpu  | freesolv      | 88.6     |              |              |               | 3.137      | 0.0          |
| conditional          | gin_ref_gpu        | tox21         | 89.9     | 87.0         | 90.1         | 1.418         |            |              |
| conditional          | gin_ref_gpu        | bbbp          | 90.0     | 91.8         | 81.6         | 1.108         |            |              |
| conditional          | gin_ref_gpu        | clintox       | 91.0     | 88.7         | 93.5         | 1.368         |            |              |
| conditional          | gin_ref_gpu        | esol          | 85.8     |              |              |               | 3.086      | 0.0          |
| conditional          | gin_ref_gpu        | lipophilicity | 88.2     |              |              |               | 2.248      | 0.0          |
| conditional          | gin_ref_gpu        | bace          | 88.2     | 90.8         | 83.5         | 1.284         |            |              |
| conditional          | gin_ref_gpu        | sider         | 91.0     | 92.1         | 91.0         | 1.747         |            |              |
| conditional          | gin_ref_gpu        | freesolv      | 84.0     |              |              |               | 3.453      | 0.0          |
| conditional          | rf                 | tox21         | 89.9     | 91.1         | 89.8         | 1.515         |            |              |
| conditional          | rf                 | bbbp          | 92.0     | 92.7         | 89.2         | 1.203         |            |              |
| conditional          | rf                 | clintox       | 90.8     | 92.2         | 93.5         | 1.628         |            |              |
| conditional          | rf                 | esol          | 87.7     |              |              |               | 5.925      | 0.0          |
| conditional          | rf                 | lipophilicity | 88.1     |              |              |               | 3.112      | 0.0          |
| conditional          | gnn                | tox21         | 89.9     | 89.6         | 89.9         | 1.472         |            |              |
| conditional          | gnn                | bbbp          | 92.3     | 92.6         | 90.1         | 1.245         |            |              |
| conditional          | gnn                | clintox       | 90.3     | 90.4         | 91.4         | 1.521         |            |              |
| conditional          | gnn                | esol          | 93.7     |              |              |               | 3.952      | 0.0          |
| conditional          | gnn                | lipophilicity | 87.6     |              |              |               | 2.432      | 0.0          |
| conditional          | trf                | tox21         | 90.0     | 90.0         | 90.0         | 1.518         |            |              |
| conditional          | trf                | bbbp          | 88.4     | 87.8         | 91.9         | 1.27          |            |              |
| conditional          | trf                | clintox       | 93.6     | 95.9         | 95.8         | 1.353         |            |              |
| conditional          | trf                | esol          | 90.5     |              |              |               | 4.897      | 0.0          |
| conditional          | trf                | lipophilicity | 89.6     |              |              |               | 3.568      | 0.0          |
| conditional          | hybrid             | tox21         | 90.1     | 90.5         | 90.1         | 1.469         |            |              |
| conditional          | hybrid             | bbbp          | 91.4     | 91.2         | 91.4         | 1.231         |            |              |
| conditional          | hybrid             | clintox       | 90.7     | 93.1         | 93.4         | 1.255         |            |              |
| conditional          | hybrid             | esol          | 93.5     |              |              |               | 4.059      | 0.0          |
| conditional          | hybrid             | lipophilicity | 89.0     |              |              |               | 2.329      | 0.0          |
| conditional          | ens                | tox21         | 89.8     | 90.3         | 89.7         | 1.443         |            |              |
| conditional          | ens                | bbbp          | 90.9     | 90.7         | 90.9         | 1.218         |            |              |
| conditional          | ens                | clintox       | 93.6     | 95.9         | 95.8         | 1.353         |            |              |
| conditional          | ens                | esol          | 94.6     |              |              |               | 4.153      | 0.0          |
| conditional          | ens                | lipophilicity | 89.1     |              |              |               | 2.341      | 0.0          |
| cqr                  | qdesc              | esol          | 87.7     |              |              |               | 2.509      | 0.821        |
| logistic_absolute    | rf                 | tox21         | 90.0     | 85.7         | 90.2         | 1.401         |            |              |
| normalized           | rf                 | tox21         | 90.7     | 9.8          | 96.6         | 0.977         |            |              |
| normalized           | rf                 | bbbp          | 90.1     | 98.2         | 58.8         | 1.056         |            |              |
| normalized           | rf                 | clintox       | 91.8     | 50.5         | 51.3         | 0.979         |            |              |
| normalized           | rf                 | esol          | 91.8     |              |              |               | 8.139      | 4.629        |
| normalized           | rf                 | lipophilicity | 90.2     |              |              |               | 4.552      | 2.973        |
| normalized           | gnn                | tox21         | 90.0     | 72.3         | 91.0         | 1.217         |            |              |
| normalized           | gnn                | bbbp          | 92.1     | 92.8         | 88.5         | 1.256         |            |              |
| normalized           | gnn                | clintox       | 92.2     | 82.0         | 82.2         | 1.097         |            |              |
| normalized           | gnn                | esol          | 93.0     |              |              |               | 7.08       | 4.01         |
| normalized           | gnn                | lipophilicity | 90.1     |              |              |               | 4.162      | 2.723        |
| normalized           | ens                | tox21         | 89.8     | 72.4         | 91.0         | 1.173         |            |              |
| normalized           | ens                | bbbp          | 90.7     | 92.6         | 84.0         | 1.153         |            |              |
| normalized           | ens                | clintox       | 92.6     | 68.5         | 68.3         | 0.943         |            |              |
| normalized           | ens                | esol          | 91.9     |              |              |               | 7.071      | 3.979        |
| normalized           | ens                | lipophilicity | 90.5     |              |              |               | 4.169      | 2.722        |
| temperature_absolute | rf                 | tox21         | 90.7     | 9.8          | 96.6         | 0.977         |            |              |
| temperature_absolute | rf                 | bbbp          | 90.1     | 98.2         | 58.8         | 1.056         |            |              |
| temperature_absolute | rf                 | clintox       | 91.8     | 50.5         | 51.3         | 0.979         |            |              |
| temperature_absolute | gnn                | tox21         | 90.0     | 72.3         | 91.0         | 1.217         |            |              |
| temperature_absolute | gnn                | bbbp          | 92.1     | 92.8         | 88.5         | 1.256         |            |              |
| temperature_absolute | gnn                | clintox       | 92.2     | 82.0         | 82.2         | 1.097         |            |              |
| temperature_absolute | ens                | tox21         | 89.8     | 72.4         | 91.0         | 1.173         |            |              |
| temperature_absolute | ens                | bbbp          | 90.7     | 92.6         | 84.0         | 1.153         |            |              |
| temperature_absolute | ens                | clintox       | 92.6     | 68.5         | 68.3         | 0.943         |            |              |
