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

#### Train bias-free

```python
cd/
python main_nodesize.py --no-bias
```

#### Evaluating saved model

```python
python evaluate_saved_models.py --checkpoint-dir outputs/nodesize_study/WL_iter=5 --node-sizes 20 --output-dir outputs/nodesize_study/WL_iter=5/evaluation_test --N 500 --k-nearest 1
```
