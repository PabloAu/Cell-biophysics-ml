# Data directory

`data/generated/` is ignored by Git. Generate the smoke dataset with:

```bash
python scripts/generate_dataset.py --config configs/smoke.yaml
```

Experimental data must not be copied here until the governance checklist in
`docs/experimental_data_governance.md` is complete.
