
## Table 3: conformal coverage at a nominal 90%

| method               | tag        | dataset       | coverage | coverage_pos | coverage_neg | mean_set_size | mean_width | width_spread |
|----------------------|------------|---------------|----------|--------------|--------------|---------------|------------|--------------|
| marginal             | rf         | tox21         | 90.7     |              |              | 0.977         |            |              |
| marginal             | rf         | bbbp          | 90.1     |              |              | 1.056         |            |              |
| marginal             | rf         | clintox       | 91.8     |              |              | 0.979         |            |              |
| marginal             | rf         | esol          | 87.7     |              |              |               | 5.925      |              |
| marginal             | rf         | lipophilicity | 88.1     |              |              |               | 3.112      |              |
| marginal             | gnn        | tox21         | 90.0     |              |              | 1.217         |            |              |
| marginal             | gnn        | bbbp          | 92.1     |              |              | 1.256         |            |              |
| marginal             | gnn        | clintox       | 92.2     |              |              | 1.097         |            |              |
| marginal             | gnn        | esol          | 93.7     |              |              |               | 3.952      |              |
| marginal             | gnn        | lipophilicity | 87.6     |              |              |               | 2.432      |              |
| marginal             | trf        | tox21         | 91.0     |              |              | 0.993         |            |              |
| marginal             | trf        | bbbp          | 85.1     |              |              | 1.315         |            |              |
| marginal             | trf        | clintox       | 92.6     |              |              | 0.943         |            |              |
| marginal             | trf        | esol          | 90.5     |              |              |               | 4.897      |              |
| marginal             | trf        | lipophilicity | 89.6     |              |              |               | 3.568      |              |
| marginal             | hybrid     | tox21         | 90.2     |              |              | 1.187         |            |              |
| marginal             | hybrid     | bbbp          | 90.9     |              |              | 1.23          |            |              |
| marginal             | hybrid     | clintox       | 90.9     |              |              | 0.994         |            |              |
| marginal             | hybrid     | esol          | 93.5     |              |              |               | 4.059      |              |
| marginal             | hybrid     | lipophilicity | 89.0     |              |              |               | 2.329      |              |
| marginal             | ens        | tox21         | 89.8     |              |              | 1.173         |            |              |
| marginal             | ens        | bbbp          | 90.7     |              |              | 1.153         |            |              |
| marginal             | ens        | clintox       | 92.6     |              |              | 0.943         |            |              |
| marginal             | ens        | esol          | 94.6     |              |              |               | 4.153      |              |
| marginal             | ens        | lipophilicity | 89.1     |              |              |               | 2.341      |              |
| absolute             | desc       | tox21         | 90.0     | 77.9         | 90.7         | 1.212         |            |              |
| absolute             | desc       | bbbp          | 90.5     | 93.1         | 79.3         | 1.057         |            |              |
| absolute             | desc       | clintox       | 90.8     | 79.2         | 82.8         | 1.139         |            |              |
| absolute             | desc       | esol          | 85.1     |              |              |               | 2.215      | 0.0          |
| absolute             | desc       | lipophilicity | 88.1     |              |              |               | 2.004      | 0.0          |
| absolute             | desc       | bace          | 85.7     | 89.6         | 81.5         | 1.201         |            |              |
| absolute             | desc       | sider         | 90.4     | 87.0         | 86.6         | 1.587         |            |              |
| absolute             | desc       | freesolv      | 90.2     |              |              |               | 2.918      | 0.0          |
| absolute             | fuse_gated | tox21         | 90.3     | 72.4         | 91.3         | 1.172         |            |              |
| absolute             | fuse_gated | bbbp          | 90.1     | 92.2         | 80.6         | 1.029         |            |              |
| absolute             | fuse_gated | clintox       | 92.4     | 89.6         | 87.1         | 0.935         |            |              |
| absolute             | fuse_gated | esol          | 82.3     |              |              |               | 2.345      | 0.0          |
| absolute             | fuse_gated | lipophilicity | 87.4     |              |              |               | 2.076      | 0.0          |
| absolute             | fuse_gated | bace          | 85.7     | 88.3         | 82.5         | 1.254         |            |              |
| absolute             | fuse_gated | sider         | 90.4     | 86.1         | 85.8         | 1.583         |            |              |
| absolute             | fuse_gated | freesolv      | 89.8     |              |              |               | 3.415      | 0.0          |
| absolute_aps         | desc       | bbbp          | 100.0    | 100.0        | 100.0        | 2.0           |            |              |
| absolute_raps        | desc       | bbbp          | 100.0    | 100.0        | 100.0        | 2.0           |            |              |
| conditional          | desc       | tox21         | 90.0     | 90.6         | 90.0         | 1.411         |            |              |
| conditional          | desc       | bbbp          | 91.3     | 92.9         | 83.9         | 1.122         |            |              |
| conditional          | desc       | clintox       | 89.5     | 87.1         | 91.7         | 1.545         |            |              |
| conditional          | desc       | bace          | 85.4     | 86.1         | 83.7         | 1.229         |            |              |
| conditional          | desc       | sider         | 91.2     | 92.7         | 89.7         | 1.725         |            |              |
| conditional          | fuse_gated | tox21         | 90.1     | 88.8         | 90.2         | 1.402         |            |              |
| conditional          | fuse_gated | bbbp          | 91.6     | 92.0         | 89.2         | 1.058         |            |              |
| conditional          | fuse_gated | clintox       | 92.2     | 94.9         | 95.3         | 1.301         |            |              |
| conditional          | fuse_gated | bace          | 89.5     | 92.2         | 85.9         | 1.334         |            |              |
| conditional          | fuse_gated | sider         | 90.6     | 90.4         | 90.6         | 1.735         |            |              |
| cqr                  | qdesc      | esol          | 87.7     |              |              |               | 2.509      | 0.821        |
| logistic_absolute    | rf         | tox21         | 90.0     | 85.7         | 90.2         | 1.401         |            |              |
| normalized           | rf         | tox21         | 90.7     | 9.8          | 96.6         | 0.977         |            |              |
| normalized           | rf         | bbbp          | 90.1     | 98.2         | 58.8         | 1.056         |            |              |
| normalized           | rf         | clintox       | 91.8     | 50.5         | 51.3         | 0.979         |            |              |
| normalized           | rf         | esol          | 91.8     |              |              |               | 8.139      | 4.629        |
| normalized           | rf         | lipophilicity | 90.2     |              |              |               | 4.552      | 2.973        |
| normalized           | gnn        | tox21         | 90.0     | 72.3         | 91.0         | 1.217         |            |              |
| normalized           | gnn        | bbbp          | 92.1     | 92.8         | 88.5         | 1.256         |            |              |
| normalized           | gnn        | clintox       | 92.2     | 82.0         | 82.2         | 1.097         |            |              |
| normalized           | gnn        | esol          | 93.0     |              |              |               | 7.08       | 4.01         |
| normalized           | gnn        | lipophilicity | 90.1     |              |              |               | 4.162      | 2.723        |
| normalized           | ens        | tox21         | 89.8     | 72.4         | 91.0         | 1.173         |            |              |
| normalized           | ens        | bbbp          | 90.7     | 92.6         | 84.0         | 1.153         |            |              |
| normalized           | ens        | clintox       | 92.6     | 68.5         | 68.3         | 0.943         |            |              |
| normalized           | ens        | esol          | 91.9     |              |              |               | 7.071      | 3.979        |
| normalized           | ens        | lipophilicity | 90.5     |              |              |               | 4.169      | 2.722        |
| temperature_absolute | rf         | tox21         | 90.7     | 9.8          | 96.6         | 0.977         |            |              |
| temperature_absolute | rf         | bbbp          | 90.1     | 98.2         | 58.8         | 1.056         |            |              |
| temperature_absolute | rf         | clintox       | 91.8     | 50.5         | 51.3         | 0.979         |            |              |
| temperature_absolute | gnn        | tox21         | 90.0     | 72.3         | 91.0         | 1.217         |            |              |
| temperature_absolute | gnn        | bbbp          | 92.1     | 92.8         | 88.5         | 1.256         |            |              |
| temperature_absolute | gnn        | clintox       | 92.2     | 82.0         | 82.2         | 1.097         |            |              |
| temperature_absolute | ens        | tox21         | 89.8     | 72.4         | 91.0         | 1.173         |            |              |
| temperature_absolute | ens        | bbbp          | 90.7     | 92.6         | 84.0         | 1.153         |            |              |
| temperature_absolute | ens        | clintox       | 92.6     | 68.5         | 68.3         | 0.943         |            |              |
