
## Table 2: paired comparisons, corrected and across datasets

| comparison                             | per-dataset | after Holm | datasets favoured | p (sign) | p (Wilcoxon, dz) | p (Wilcoxon, raw) |
|----------------------------------------|-------------|------------|-------------------|----------|------------------|-------------------|
| fuse_proposed vs gin_ref               | 4+ / 0-     | 2+ / 0-    | 8/8               | 0.0078   | 0.0078           | 0.0078            |
| fuse_proposed_e2e vs fuse_concat_e2e   | 4+ / 0-     | 1+ / 0-    | 8/8               | 0.0078   | 0.0078           | 0.0078            |
| fuse_proposed_e2e vs fuse_xattn_e2e    | 3+ / 0-     | 2+ / 0-    | 7/8               | 0.0703   | 0.0156           | 0.0156            |
| attentivefp vs gine                    | 1+ / 0-     | 0+ / 0-    | 7/8               | 0.0703   | 0.0391           | 0.0234            |
| fuse_proposed vs fuse_concat           | 2+ / 0-     | 1+ / 0-    | 7/8               | 0.0703   | 0.0391           | 0.0234            |
| fuse_bilinear vs fuse_concat           | 0+ / 0-     | 0+ / 0-    | 7/8               | 0.0703   | 0.0391           | 0.1953            |
| desc vs gin_ref                        | 1+ / 0-     | 0+ / 0-    | 7/8               | 0.0703   | 0.0547           | 0.0781            |
| fuse_proposed_gpu vs attentivefp       | 2+ / 0-     | 0+ / 0-    | 7/8               | 0.0703   | 0.0547           | 0.0547            |
| fuse_proposed vs fuse_xattn            | 1+ / 0-     | 0+ / 0-    | 6/8               | 0.2891   | 0.0781           | 0.1094            |
| fuse_bilinear_e2e vs fuse_concat_e2e   | 3+ / 0-     | 1+ / 0-    | 6/8               | 0.2891   | 0.0781           | 0.1484            |
| chemprop vs desc                       | 0+ / 2-     | 0+ / 1-    | 2/8               | 0.2891   | 0.1094           | 0.3828            |
| fuse_proposed_gpu vs chemprop          | 3+ / 0-     | 2+ / 0-    | 6/8               | 0.2891   | 0.1094           | 0.1953            |
| attentivefp vs desc                    | 0+ / 1-     | 0+ / 0-    | 2/8               | 0.2891   | 0.1484           | 0.3125            |
| fuse_proposed_gpu vs fuse_proposed     | 0+ / 0-     | 0+ / 0-    | 3/8               | 0.7266   | 0.1484           | 0.1484            |
| fuse_proposed_e2e vs fuse_bilinear_e2e | 0+ / 0-     | 0+ / 0-    | 6/8               | 0.2891   | 0.1484           | 0.2500            |
| fuse_proposed_e2e vs fuse_proposed     | 0+ / 0-     | 0+ / 0-    | 1/8               | 0.0703   | 0.1484           | 0.1094            |
| attentivefp vs gin_ref_gpu             | 1+ / 0-     | 0+ / 0-    | 6/8               | 0.2891   | 0.1953           | 0.1094            |
| fuse_proposed vs fuse_gated            | 1+ / 0-     | 0+ / 0-    | 5/8               | 0.7266   | 0.2500           | 0.1953            |
| lora vs seq_frozen                     | 2+ / 0-     | 1+ / 0-    | 5/8               | 0.7266   | 0.2500           | 0.3125            |
| fuse_proposed_e2e vs fuse_gated_e2e    | 2+ / 0-     | 2+ / 0-    | 5/8               | 0.7266   | 0.3125           | 0.3125            |
| fuse_proposed vs fuse_bilinear         | 1+ / 0-     | 0+ / 0-    | 5/8               | 0.7266   | 0.3125           | 0.1484            |
| fuse_xattn_e2e vs fuse_concat_e2e      | 1+ / 0-     | 1+ / 0-    | 4/8               | 1.0000   | 0.3828           | 0.5469            |
| fuse_concat_gpu vs fuse_concat         | 0+ / 0-     | 0+ / 0-    | 2/8               | 0.2891   | 0.3828           | 0.3125            |
| fuse_xattn_e2e vs fuse_gated_e2e       | 1+ / 1-     | 0+ / 0-    | 3/8               | 0.7266   | 0.4609           | 0.4609            |
| seq_frozen vs gin_ref                  | 2+ / 4-     | 1+ / 1-    | 2/8               | 0.2891   | 0.4609           | 0.2500            |
| fuse_proposed vs desc                  | 2+ / 0-     | 0+ / 0-    | 4/8               | 1.0000   | 0.5469           | 0.4609            |
| fuse_concat vs desc                    | 1+ / 1-     | 0+ / 0-    | 2/8               | 0.2891   | 0.5469           | 0.5469            |
| gin_ref_gpu vs gin_ref                 | 0+ / 0-     | 0+ / 0-    | 3/8               | 0.7266   | 0.5469           | 0.5469            |
| lora vs gin_ref                        | 2+ / 4-     | 1+ / 1-    | 3/8               | 0.7266   | 0.5469           | 0.3828            |
| fuse_gated_nograph vs fuse_gated       | 1+ / 0-     | 0+ / 0-    | 4/8               | 1.0000   | 0.6406           | 0.6406            |
| fuse_xattn vs fuse_concat              | 0+ / 0-     | 0+ / 0-    | 5/8               | 0.7266   | 0.6406           | 0.4609            |
| fuse_gated vs desc                     | 2+ / 0-     | 0+ / 0-    | 2/8               | 0.2891   | 0.7422           | 0.5469            |
| fuse_bilinear_gpu vs fuse_bilinear     | 0+ / 0-     | 0+ / 0-    | 3/8               | 0.7266   | 0.8438           | 0.7422            |
| fuse_proposed_e2e vs desc              | 2+ / 1-     | 0+ / 0-    | 3/8               | 0.7266   | 0.8438           | 0.7422            |
| chemprop vs gin_ref_gpu                | 0+ / 0-     | 0+ / 0-    | 4/8               | 1.0000   | 0.9453           | 0.8438            |
| fuse_bilinear vs desc                  | 2+ / 0-     | 1+ / 0-    | 3/8               | 0.7266   | 0.9453           | 0.7422            |
| fuse_xattn vs desc                     | 2+ / 2-     | 0+ / 0-    | 4/8               | 1.0000   | 0.9453           | 0.9453            |
| fuse_bilinear_e2e vs fuse_gated_e2e    | 1+ / 0-     | 0+ / 0-    | 3/8               | 0.7266   | 0.9453           | 0.7422            |
| chemprop vs attentivefp                | 0+ / 2-     | 0+ / 0-    | 4/8               | 1.0000   | 1.0000           | 1.0000            |
| fuse_xattn_gpu vs fuse_xattn           | 0+ / 0-     | 0+ / 0-    | 4/8               | 1.0000   | 1.0000           | 0.9453            |
| gine vs gin_ref                        | 1+ / 1-     | 0+ / 0-    | 5/8               | 0.7266   | 1.0000           | 0.7422            |
