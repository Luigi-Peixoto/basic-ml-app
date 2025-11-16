# data

As of now, the training data used by `intent-classifier` should be in yml format. 

For example:
```yml
- intent: confusion
  examples:
    - wait what?
    - huh? im confused
    ...

- intent: neutral
  examples:
    - Alright, let's see what happens
    - I'm not ready
    - We should continue with the next part
    ...
```

## How to train?

```bash
python intent_classifier.py train \
    --config="models/confusion-v1_config.yml" \
    --examples_file="data/confusion_intents.yml" \
    --save_model="models/confusion-v1.keras"
```
