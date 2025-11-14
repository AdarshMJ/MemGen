#### Generate training data

```python
python seeded_graph_generator.py \
  --output-dir data/diverse_splits \
  --node-sizes 20 50 100 500 600 700 800 \
  --train-per-set 1000 \
  --test-count 100 \
  --diverse \
  --s1-hom-range 0.65 0.95 \
  --s2-hom-range 0.15 0.35 \
  --s1-templates 0 \
  --s2-templates 1 \
  --validate-diversity \
  --diversity-margin 0.12 \
  --visualize
```

#### Train bias-free - Trains VGAE+LDM model

```python
cd/Updated
python main_nodesize.py --no-bias
```
 Evaluating saved model

```python
python evaluate_saved_models.py --checkpoint-dir outputs/nodesize_study/WL_iter=5 --node-sizes 20 --output-dir outputs/nodesize_study/WL_iter=5/evaluation_test --N 500 --k-nearest 1
```

#### Train GraphMaker on own dataset
```python
   python lightweight_graphmaker/experiment.py \
    --node_sizes 20 50 100 \
    --num_generated 100
```

#### Jacobian

```python
  python run_jacobian_analysis.py \
    --checkpoint experiments/n100_20251114_121458/checkpoints/DF1_n100_best.pt \
    --node_size 100 \
    --split S1 \
    --num_samples 50 \
    --output_dir jacobian_analysis/DF1_n100
```
