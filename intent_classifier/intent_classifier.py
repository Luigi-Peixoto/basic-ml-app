"""
This script works as a module and as a CLI tool.

To use it as a module, you can do:
::

    from intent_classifier import IntentClassifier

    # train a model
    classifier = IntentClassifier(config="models/confusion_config.yml", training_data="data/confusion_intents.yml")
    classifier.train(save_model="models/confusion-clf/")
    # or load a model from W&B
    classifier = IntentClassifier(config="models/confusion_config.yml", load_model="adaj/intent-classifier-2025-2/confusion-clf:v1")
    # predict a new text
    classifier.predict(input_text="oi")
    # cross-validate the model
    classifier.cross_validation(n_splits=5)

Or, or you can use it as a CLI tool, you can do:
::

cd intent_classifier

python intent_classifier.py train \
    --config="models/confusion_config.yml" \
    --training_data="data/confusion_intents.yml" \
    --save_model="models/confusion.keras" \
    --wandb_project="intent-classifier"

python intent_classifier.py predict \
    --load_model="models/confusion.keras" \
    --input_text="teste teste" \
    --wandb_project="intent-classifier"

# TODO: Fix CV implementation...
python intent_classifier.py cross_validation \
    --config="models/confusion_config.yml" \
    --training_data="data/confusion_intents.yml" \
    --n_splits=5 \
    --wandb_project="intent-classifier"

"""
# instalar alguns pacotes auxiliares

import os
import logging
from pathlib import Path
from typing import List, Optional, Union, Tuple, Dict, Any
from dataclasses import dataclass
import yaml
from pprint import pprint
import re

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, cohen_kappa_score

import tensorflow as tf
from tensorflow.keras import regularizers
import tensorflow_text
import tensorflow_hub as hub
from tensorflow.keras.saving import register_keras_serializable

import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbEvalCallback # WandbModelCheckpoint

import dotenv
dotenv.load_dotenv()

logger = logging.getLogger(__name__)

@register_keras_serializable()
class HubLayer(tf.keras.layers.Layer):
    """
    A custom Keras layer to load and use a TensorFlow Hub module.

    This layer loads a pre-trained model from a TensorFlow Hub URL
    and integrates it into a Keras model. It can be set to be
    trainable or frozen.

    :param hub_url: The URL of the TensorFlow Hub module to load.
    :type hub_url: str
    :param trainable: Whether the loaded Hub module should be trainable.
    :type trainable: bool, optional
    """
    def __init__(self, hub_url, trainable=False, **kwargs):
        """
        Initializes the HubLayer.
        """
        super(HubLayer, self).__init__(**kwargs)
        self.hub_module = hub.load(hub_url)
        self.hub_module.trainable = trainable

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Executes the forward pass of the layer.

        :param inputs: The input tensor(s) to the Hub module.
        :type inputs: tf.Tensor
        :return: The output tensor(s) from the Hub module.
        :rtype: tf.Tensor
        """
        return self.hub_module(inputs)

@dataclass
class Config:
    """
    A dataclass to hold all configuration parameters for the IntentClassifier.

    This object stores settings related to the dataset, model architecture,
    training process, and logging.
    """
    dataset_name: str = "undefined"
    """Name of the dataset, used for logging and model naming."""
    codes : List[str] = None
    """A list of intent codes (class labels). Automatically populated from data if not provided."""
    architecture: str = "v0.1.5"
    """Version tag for the model architecture."""
    task: str = "undefined"
    """The current task being performed (e.g., 'train', 'predict')."""
    stop_words_file: Optional[str] = None
    """Path to a text file containing stopwords, one per line."""
    min_words: int = 1
    """The minimum number of words required in an utterance for processing. Shorter inputs are padded."""
    embedding_model: Union[str, List[str]] = 'https://www.kaggle.com/models/google/universal-sentence-encoder/tensorFlow2/multilingual/2'
    """URL or path to the TensorFlow Hub embedding model."""
    sent_hl_units: Union[int, List[int]] = 32
    """Number of units in the hidden layer."""
    sent_dropout: Union[float, List[float]] = 0.1
    """Dropout rate applied after the hidden layer."""
    l1_reg: float = 0.01
    """L1 regularization factor for the hidden layer kernel."""
    l2_reg: float = 0.01
    """L2 regularization factor for the hidden layer kernel."""
    epochs: int = 500
    """Maximum number of epochs for training."""
    callback_patience: int = 20
    """Number of epochs with no improvement to wait before early stopping."""
    learning_rate: Union[float, List[float]] = 5e-3
    """Initial learning rate for the optimizer."""
    validation_split: float = 0.2
    """Fraction of the training data to be used as validation data."""

def remove_duplicate_words(text: str) -> str:
    """
    Removes consecutive duplicate words from a string.

    :param text: The input string.
    :type text: str
    :return: The text with consecutive duplicates removed.
    :rtype: str
    """
    # Note: The original implementation removes ALL duplicate words, not just consecutive.
    # The docstring reflects the implementation.
    words = text.split()
    seen = set()
    result = []
    for word in words:
        if word not in seen:
            seen.add(word)
            result.append(word)
    return ' '.join(result)


def fetch_artifact_from_wandb(model_full_name: str) -> Tuple[str, str]:
    """
    Download a model artifact from W&B and return the paths to the model and config files.

    :param model_full_name: The W&B artifact full name (e.g., "adaj/intent-classifier-2025-2/confusion-clf:v1").
                           Must have format: "entity/project/artifact_name:version"
    :type model_full_name: str
    :return: A tuple containing the local file path to the Keras model file and the config file.
    :rtype: tuple[str, str]
    :raises ValueError: If format is invalid or files are not found in the artifact.
    """
    # Validate format
    parts = model_full_name.split("/")
    if len(parts) != 3 or ":" not in parts[2]:
        raise ValueError(
            f"Invalid model_full_name format: '{model_full_name}'. "
            f"Expected format: 'entity/project/artifact_name:version' (e.g., 'adaj/intent-classifier-2025-2/confusion-clf:v1')"
        )
    
    # Download artifact from W&B
    try:
        api = wandb.Api()
        artifact = api.artifact(model_full_name, type='model')
    except wandb.errors.CommError as e:
        raise ValueError(f"Could not fetch artifact '{model_full_name}' from W&B. Ensure the path is correct and you are logged in. Original error: {e}")

    # Create a target directory for the download
    models_dir = Path(os.path.dirname(__file__)) / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Download artifact content. The path returned is the directory where files are.
    download_path = artifact.download(root=models_dir)
    
    model_file, config_file = None, None
    # Iterate over the files *in the artifact manifest* to find the correct ones.
    # This prevents accidentally loading unrelated files from the same directory.
    for f in artifact.files():
        if f.name.endswith((".keras", ".h5")):
            model_file = os.path.join(download_path, f.name)
        elif f.name.endswith("_config.yml"):
            config_file = os.path.join(download_path, f.name)
            
    if not model_file:
        raise ValueError(f"Model file (.keras or .h5) not found in W&B artifact '{model_full_name}'.")
    if not config_file:
        raise ValueError(f"Config file (_config.yml) not found in W&B artifact '{model_full_name}'.")
        
    return model_file, config_file


class IntentClassifier:
    """
    A class for training, evaluating, and predicting text intents using a Keras model.

    This classifier wraps the entire MLOps pipeline, including data loading,
    preprocessing, model building (using TensorFlow Hub embeddings), training,
    W&B logging, and prediction.

    :param config: A path to a YAML config file, a Config object, or None.
                   If None, config is inferred from `load_model`.
    :type config: str, Config, optional
    :param load_model: A path to a saved Keras model (`.keras` file) or a W&B artifact URL.
                       If provided, the model and its associated config are loaded.
    :type load_model: str, optional
    :param training_data: Path to a YAML file containing training examples.
                          Required for training or cross-validation.
    :type training_data: str, optional
    :raises ValueError: If `config` is not provided and cannot be inferred from `load_model`.
    :raises ValueError: If `load_model` path is provided but the file is not found.
    """

    def __init__(self, config: Optional[Union[str, Config]] = None,
                 load_model: Optional[str] = None,
                 training_data: Optional[str] = None,
                 wandb_project: Optional[str] = None):
        """
        Initializes the IntentClassifier.
        """
        self.model = None
        local_model_path = None
        
        # Set up W&B project early
        self.wandb_project = wandb_project or os.environ.get("WANDB_PROJECT") or "intent-classifier"

        if load_model:
            # Check if load_model is a local file path or a W&B artifact URL
            if os.path.exists(load_model):
                local_model_path = load_model
            else:
                # If not a local path, assume it's a W&B artifact and fetch it.
                # The associated config path will be discovered and used automatically.
                local_model_path, config = fetch_artifact_from_wandb(load_model)
            
            self.model = tf.keras.models.load_model(local_model_path)
            print(f"Loaded Keras model from {local_model_path}.")

        # Load config. If fetched from W&B, `config` is already the correct path.
        self._load_config(config)
        
        # Load intents from the examples file if provided
        self._load_intents(training_data)
        
        # Validate model output size matches config codes (if model is loaded)
        if self.model:
            self._validate_model_config_compatibility()
            
        # Initialize stop_words and one-hot encoder
        self._load_stop_words(self.config.stop_words_file)
        self._setup_onehot_encoder()
        
        # Set up W&B run
        if self.wandb_project:
            print(f"Setting up W&B project: {self.wandb_project}")
            wandb.login(key=os.environ.get("WANDB_API_KEY"))
            self.wandb_run = wandb.init(project=self.wandb_project, config=self.config.__dict__)
            if self.training_data:
                artifact = wandb.Artifact(Path(self.training_data).name, type="dataset")
                artifact.add_file(self.training_data)
                self.wandb_run.log_artifact(artifact)
        else:
            self.wandb_run = None
            print("W&B project not set. No W&B run will be created.")

    def _load_config(self, config: Optional[Union[str, Config]]) -> None:
        """
        Loads the configuration from a file path or a Config object.

        :param config: A path to a YAML config file or a Config object.
        :type config: str, Config, optional
        :raises ValueError: If config is not provided or is of an invalid type.
        """
        if isinstance(config, str):
            with open(config, 'r') as f:
                self.config = Config(**yaml.safe_load(f))
            print(f"Loaded config from {config}.")
        elif isinstance(config, Config):
            self.config = config
        elif config is None:
            # If no config is provided at all, we can't proceed.
            raise ValueError(
                "A 'config' must be provided, either as a file path, a Config object, "
                "or as part of a W&B artifact."
            )
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")
    
    def _load_intents(self, training_data: Optional[str]) -> np.ndarray:
        """
        Loads and preprocesses intents from a YAML examples file.

        If `training_data` is provided, it reads the file, extracts utterances
        and labels, shuffles them, and stores them in `self.input_text`
        and `self.labels`. It also populates `self.codes` and `self.config.codes`.

        If `training_data` is None, it loads `self.codes` from `self.config.codes`.

        :param training_data: Path to a YAML file with intent examples.
        :type training_data: str, optional
        :return: An array of unique intent codes (labels).
        :rtype: np.ndarray
        """
        self.training_data = training_data
        if training_data is not None:
            pprint(f"Loading intents from {training_data}...")
            with open(training_data, 'r') as f:
                self.intents_data = yaml.safe_load(f)
            # Preprocess intents
            input_text = []
            labels = []
            for i in self.intents_data:
                input_text += i['examples']
                labels += [i['intent']]*len(i['examples'])
            input_text = np.array(input_text)
            labels = np.array(labels)
            # Shuffle data
            indices = np.arange(len(labels))
            np.random.shuffle(indices)
            self.input_text = input_text[indices]
            self.input_text = tf.convert_to_tensor(self.input_text, dtype=tf.string)
            self.labels = labels[indices]
            self.codes = np.unique(self.labels)
            self.config.codes = self.codes.tolist()
        else: # Means that the example_file is not provided
            # Then the model will be used only to predict, no need to load training data
            self.codes = self.config.codes
        return self.codes
    
    def _load_stop_words(self, stop_words_file: Optional[str]) -> 'IntentClassifier':
        """
        Loads stopwords from a file.

        :param stop_words_file: Path to a text file containing stopwords, one per line.
        :type stop_words_file: str, optional
        :return: The IntentClassifier instance.
        :rtype: IntentClassifier
        """
        if stop_words_file is None:
            self.stop_words = []
            return self
        with open(stop_words_file, 'r', encoding='utf-8') as f:
            self.stop_words = f.read().split('\n')
        print(f"Loaded {len(self.stop_words)} stop words from {stop_words_file}.")
        return self
    
    def _validate_model_config_compatibility(self) -> None:
        """
        Validates that the loaded model's output size matches the number of categories
        in config.codes. This ensures the model and config are compatible.

        :raises ValueError: If the model's output size doesn't match the number of codes
                            in the config, indicating an invalid model-config mismatch.
        """
        if self.model is None:
            return
        # Get the model's output layer size (number of classes the model was trained with)
        model_output_size = self.model.output_shape[-1]
        # Get the number of codes from config
        config_codes_count = len(self.config.codes) if self.config.codes else 0
        # Also check onehot_encoder categories if it exists (though it shouldn't at this point)
        # This is a double-check to ensure consistency
        if model_output_size != config_codes_count:
            error_msg = (
                f"Model-config mismatch detected: The loaded model was trained with "
                f"{model_output_size} categories, but the config file specifies "
                f"{config_codes_count} categories (codes: {self.config.codes}).\n\n"
                f"This indicates an invalid model configuration. The model's output layer "
                f"and the one-hot encoder categories must match.\n\n"
            )
            raise ValueError(error_msg)
    
    def _setup_onehot_encoder(self) -> OneHotEncoder:
        """
        Initializes and fits the OneHotEncoder based on the loaded intent codes.

        :return: The fitted OneHotEncoder instance.
        :rtype: OneHotEncoder
        """
        assert self.codes is not None, "codes must be set before setting up the encoder."
        self.onehot_encoder = OneHotEncoder(categories=[self.codes],)\
                                  .fit(np.array(self.codes).reshape(-1, 1))
        return self.onehot_encoder

    def _get_callbacks(self) -> list:
        """
        Constructs a list of Keras callbacks based on the configuration.

        Includes EarlyStopping and WandbMetricsLogger if configured.
        Also includes an ExponentialDecay learning rate scheduler.

        :return: A list of Keras Callback instances.
        :rtype: list
        """
        callbacks = []
        if self.config.callback_patience > 0:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(monitor='val_f1_score',
                    patience=self.config.callback_patience,
                    restore_best_weights=True)
            )
        if self.wandb_project:
            callbacks.append(WandbMetricsLogger())
        
        # Configure ExponentialDecay
        if self.config.learning_rate is not None and not isinstance(self.config.learning_rate, str):
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.config.learning_rate,
                decay_steps=1000,
                decay_rate=0.96,
                staircase=False
            )
            
            # Modified learning rate scheduler to properly handle epoch parameter
            def lr_scheduler(epoch, lr):
                """Internal LR scheduler function."""
                return lr_schedule(epoch).numpy().astype(float)
            
            lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
            callbacks.append(lr_scheduler_callback)
        return callbacks
    
    def finish_wandb(self):
        """
        Finishes the current Weights & Biases run, if one is active.
        """
        if self.wandb_project and self.wandb_run:
            self.wandb_run.finish()

    def preprocess_text(self, text: tf.Tensor) -> tf.Tensor:
        """
        Applies preprocessing steps to a single raw text tensor.

        Steps include:
        1. Lowercasing.
        2. Stopword removal (if configured).
        3. Padding with "<>" tokens if text is shorter than `min_words`.

        :param text: A 0-D string tensor (scalar) containing the raw text.
        :type text: tf.Tensor
        :return: A 0-D string tensor (scalar) containing the preprocessed text.
        :rtype: tf.Tensor
        """
        text = tf.strings.lower(text)
        if self.stop_words:
            words = tf.strings.split(text)
            # Create a mask for words that are NOT in stopwords
            words = tf.boolean_mask(words, tf.reduce_all(tf.not_equal(words[:, None], tf.constant(self.stop_words)), axis=1))
            text = tf.strings.reduce_join(words, separator=' ')
        
        if self.config.min_words:
            # Remove punctuation before counting words
            words = tf.strings.split(text)
            words = tf.boolean_mask(words, tf.reduce_all(tf.not_equal(words[:, None], tf.constant(["?", ".", ",", "!"])), axis=1))
            num_words = tf.shape(words)[0]
            
            # Check if num_words is less than or equal to min_words
            if tf.less_equal(num_words, self.config.min_words):
                # Create padding
                padding = tf.strings.join(["<>"] * (self.config.min_words + 1), separator=' ')
                text = padding

        # Iterate on text and replace punctuation with "PUNCTUATION" (it helps some sentence encoders that do not parse punctuation, like Universal Sentence Encoder)
        for p, t in {"?": "QUESTION_MARK", ".": "PERIOD", ",": "COMMA", "!": "EXCLAMATION_MARK"}.items():
            espaped_p = re.escape(p)
            text = tf.strings.regex_replace(text, espaped_p, f" {t} ")
        text = tf.strings.regex_replace(text, r"\s+", " ")
        text = tf.strings.strip(text)
        
        # Ensure output is always a 0-D tensor (scalar)
        return tf.strings.as_string(text)

    def make_model(self, config: Config) -> tf.keras.Model:
        """
        Builds and returns a new Keras model based on the provided configuration.

        The architecture consists of:
        1. An input layer for string tensors.
        2. A `HubLayer` for text embedding.
        3. A Dense hidden layer with BatchNormalization, ReLU activation, and Dropout.
        4. A final Dense output layer with softmax activation for classification.

        :param config: The configuration object specifying model hyperparameters.
        :type config: Config
        :return: A compiled Keras model.
        :rtype: tf.keras.Model
        """
        # Set the random seed for reproducibility
        seed = 42
        tf.random.set_seed(seed)  # Assuming you have a random_seed in your config
        # Extract config values
        sent_hl_units, sent_dropout = config.sent_hl_units, config.sent_dropout
        l1_reg, l2_reg = config.l1_reg, config.l2_reg
        output_size = len(self.codes)
        # Build model
        initializer = tf.keras.initializers.GlorotUniform(seed=seed)  # Set seed in initializer
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="inputs")
        # Sentence encoder
        encoder = HubLayer(config.embedding_model, trainable=False, name="sent_encoder")(text_input)
        # Hidden layer
        sent_hl = tf.keras.layers.Dense(sent_hl_units,
                                        kernel_initializer=initializer,
                                        kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                                        activation=None,  # No activation here yet
                                        name='sent_hl')(encoder)
        sent_hl_norm = tf.keras.layers.BatchNormalization()(sent_hl)  # Add batch normalization
        sent_hl_activation = tf.keras.layers.Activation('relu')(sent_hl_norm)  # Activation after batch normalization
        sent_hl_dropout = tf.keras.layers.Dropout(sent_dropout, seed=seed)(sent_hl_activation)  # Set seed in dropout
        # Output layer
        sent_output = tf.keras.layers.Dense(output_size,
                                            kernel_initializer=initializer,
                                            activation='softmax',
                                            name="sent_output")(sent_hl_dropout)
        model = tf.keras.Model(inputs=text_input, outputs=sent_output)
        return model

    def train(self, save_model: Optional[str] = None, tf_verbosity: int = 1) -> tf.keras.Model:
        """
        Trains the model on the loaded examples.

        This method splits the data, preprocesses it, builds a new model,
        compiles it, and runs the training loop. If `save_model` is provided,
        the trained model and its config are saved.

        :param save_model: Path to save the trained model (`.keras` format).
                           Config will be saved alongside (e.g., `model_config.yml`).
        :type save_model: str, optional
        :param tf_verbosity: Verbosity level for Keras `fit` method (0, 1, or 2).
        :type tf_verbosity: int, optional
        :return: The trained Keras model.
        :rtype: tf.keras.Model
        :raises AssertionError: If `training_data` was not provided during initialization.
        """
        pprint(self.config.__dict__)
        # Update task config parameter
        self.config.task = "train"
        assert self.training_data is not None, "training_data must be provided when the IntentClassifier was created."
        
        # Extract one-hot encoded labels
        labels_ohe = self.onehot_encoder\
                            .transform(self.labels.reshape(-1, 1))\
                            .toarray()
        # Split
        X_train_text, X_val_text, y_train, y_val = train_test_split(
            self.input_text.numpy(), labels_ohe, # Convert to NumPy array for splitting
            test_size=self.config.validation_split,
            stratify=labels_ohe,      # Ensure class distribution is preserved
            random_state=42           # For reproducibility
        )
        # Now apply preprocessing using preprocess_text *after* splitting:
        X_train = tf.map_fn(self.preprocess_text, tf.constant(X_train_text), dtype=tf.string)
        X_val = tf.map_fn(self.preprocess_text, tf.constant(X_val_text), dtype=tf.string)

        # Extract config values
        epochs = self.config.epochs
        # New model from scratch
        self.model = self.make_model(self.config)
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(), # LR is handled by callback
            metrics=[tf.keras.metrics.F1Score(average='macro')])
        # Train the model
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            # batch_size=16,
            shuffle=True,
            epochs=epochs,
            verbose=tf_verbosity,
            callbacks=self._get_callbacks()
        )
        # Save model
        if save_model is not None:
            self.save_model(path=save_model)
        return self.model

    def save_model(self, path: str):
        """
        Saves the current model and its configuration file.

        The model is saved in Keras format (`.keras`).
        The config is saved as a YAML file with `_config.yml` suffix.
        If W&B is configured, the model is also logged as an artifact.

        :param path: The base path to save the model (e.g., "models/my_model.keras").
        :type path: str
        """
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        # Save model in SavedModel format
        if path.endswith('/'):
            # Remove trailing slash if present
            path = path.rstrip('/')
        # Save the model
        self.model.save(path)
        # Save config into a yaml file inside the model directory
        config_path = path.replace(".keras", "_config.yml") #os.path.join(os.path.dirname(path), f"{self.config.dataset_name}_config.yml")
        with open(config_path, 'w') as f:
            f.write(yaml.dump(self.config.__dict__))
        print(f"Model saved to {path}.")
        if self.wandb_project:
            # Crie e envie o artifact
            artifact = wandb.Artifact(
                name=f"{self.config.dataset_name}-clf",
                type="model",
                description="Modelo Keras para classificação de intenção"
            )
            artifact.add_file(path)
            artifact.add_file(config_path) # Also add the config file
            self.wandb_run.log_artifact(artifact)
            self.finish_wandb() # Finish the run after saving

    def predict(self, input_text: Union[str, List[str]],
                true_labels: Optional[List[str]] = None,
                log_to_wandb: bool = False) -> Union[Tuple[str, Dict[str, float]], List[Tuple[str, Dict[str, float]]]]:
        """
        Predicts the intent for a given text or list of texts.

        :param input_text: A single text string or a list of text strings to classify.
        :type input_text: str or list[str]
        :param true_labels: A list of true labels, corresponding to `input_text`. Used for logging to W&B.
        :type true_labels: list[str], optional
        :param log_to_wandb: If True, logs the inputs, predictions, and true labels (if provided) to W&B.
        :type log_to_wandb: bool, optional
        :return: If `input_text` is a string: a tuple `(top_intent, all_probabilities)`.
                 If `input_text` is a list: a list of tuples `[(top_intent, all_probabilities), ...]`.
                 `all_probabilities` is a dict mapping intent codes to their predicted probabilities.
        :rtype: tuple(str, dict(str, float)) or list[tuple(str, dict(str, float))]
        """
        self.config.task = "predict"  # Set the task to "predict"
        original_input_is_string = isinstance(input_text, str)
        if original_input_is_string:
            input_text_list = [input_text]  # Convert single string to a list for processing
        else:
            input_text_list = input_text
        
        # Preprocess each string in the list and stack them
        preprocessed_texts = tf.map_fn(self.preprocess_text, tf.constant(input_text_list), dtype=tf.string)

        # Predict probabilities for all strings at once
        all_probs = self.model.predict(preprocessed_texts)
        results = []
        predicted_labels_for_log = []
        for i in range(all_probs.shape[0]):
            current_probs = all_probs[i] # Probabilities for the i-th input text
            # Determine the intent name with the highest probability
            highest_prob_idx = np.argmax(current_probs)
            highest_prob_intent_name = self.codes[highest_prob_idx]
            predicted_labels_for_log.append(highest_prob_intent_name)
            # Create a dictionary of probabilities for each intent name
            probs_dict = {code: float(current_probs[j]) for j, code in enumerate(self.codes)}
            results.append((highest_prob_intent_name, probs_dict))
        
        # Log to Wandb if requested
        if log_to_wandb and self.wandb_project:
            # Get the current run ID if it exists, otherwise start a new run
            run_id = wandb.run.id if wandb.run else wandb.util.generate_id()
            # Initialize wandb with the run ID
            with wandb.init(project=self.wandb_project, id=run_id, resume="allow"):
                wandb.log({
                    "inputs": input_text_list, # Log the list of original input texts
                    "true_labels": true_labels,
                    "predictions": predicted_labels_for_log # Use the extracted list of highest prob intents
                })
        
        # Return a single tuple if the original input was a string, otherwise a list of tuples
        if original_input_is_string:
            return results[0]
        return results

    def cross_validation(self, n_splits: int = 3) -> List[Dict[str, Any]]:
        """
        Performs stratified K-fold cross-validation.

        This method trains and evaluates the model `n_splits` times on different
        folds of the data. Metrics for each fold and the average metrics are
        logged to Weights & Biases if configured.

        :param n_splits: The number of folds to use for cross-validation.
        :type n_splits: int, optional
        :return: A list of dictionaries, where each dictionary is the classification
                 report (from `sklearn.metrics.classification_report`) for a fold.
        :rtype: list[dict(str, Any)]
        :raises AssertionError: If `training_data` was not provided during initialization.
        """
        assert self.training_data is not None, "training_data must be provided when the IntentClassifier was created."
        # Update task config parameter
        self.config.task = "cross_validation"
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Preprocess the entire dataset once
        preprocessed_input_text = tf.map_fn(self.preprocess_text, self.input_text, dtype=tf.string)
        
        # Get one-hot encoded labels before the loop
        labels_ohe = self.onehot_encoder.transform(self.labels.reshape(-1, 1)).toarray()
        
        results = []
        
        for i, (train_index, test_index) in enumerate(kf.split(preprocessed_input_text.numpy(), self.labels)):
            print(f"Fold {i+1}/{n_splits}")
            # Create and log a new Wandb run for each fold
            run_name = f"cv_fold_{i+1}"
            with wandb.init(project=self.wandb_project, config=self.config.__dict__, 
                            group="cross_validation", name=run_name, reinit=True, 
                            job_type=f"fold_{i+1}"):
                
                # Create a new model for each fold
                model = self.make_model(self.config)
                model.compile(
                    loss='categorical_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(), # LR handled by callback
                    metrics=[tf.keras.metrics.F1Score(average='macro')])
                
                # Get train/test splits for this fold
                X_train, X_test = preprocessed_input_text[train_index], preprocessed_input_text[test_index]
                y_train_ohe, y_test_ohe = labels_ohe[train_index], labels_ohe[test_index]

                # Train the model on the current fold
                model.fit(X_train, y_train_ohe,
                          epochs=self.config.epochs, verbose=0,
                          validation_data=(X_test, y_test_ohe), # Use test fold as validation
                          callbacks=self._get_callbacks()) # WandbMetricsLogger is already added in _get_callbacks()
                
                # Predict on the test set for the current fold
                preds_probs = model.predict(X_test)
                preds = self.onehot_encoder.inverse_transform(preds_probs)
                labels = self.onehot_encoder.inverse_transform(y_test_ohe)
                
                # Evaluate the model and store the results
                res = classification_report(labels, preds, output_dict=True, zero_division=0)
                res['kappa'] = cohen_kappa_score(labels, preds)
                results.append(res)
                
                # Log fold-specific metrics
                wandb.log({"fold_results": res, "val_f1_macro": res["macro avg"]["f1-score"], "val_kappa": res['kappa']})
        
        # Calculate and print average metrics
        avg_f1 = np.mean([r['macro avg']['f1-score'] for r in results])
        avg_kappa = np.mean([r['kappa'] for r in results])
        print(f"Average f1-score: {avg_f1}")
        print(f"Average kappa: {avg_kappa}")
        
        # Log average metrics to a summary run
        with wandb.init(project=self.wandb_project, config=self.config.__dict__, 
                        group="cross_validation", name="cv_summary", reinit=True, 
                        job_type="summary"):
            wandb.log({"avg_f1_macro": avg_f1, "avg_kappa": avg_kappa})
            
        self.finish_wandb() # Finish the last summary run
        return results


# This script works as a module and as a CLI tool
if __name__ == "__main__":
    import fire
    # Instead of fire.Fire(IntentClassifier),
    # Define the functions to be used by Fire CLI so that 
    #  it's not cluttered with all the functions in the IntentClassifier class
    def train(config: str, training_data: str, save_model: str, wandb_project: str = None):
        """
        Train the model with the given configuration and examples.

        :param config: Path to the YAML configuration file.
        :type config: str
        :param training_data: Path to the YAML file with training examples.
        :type training_data: str
        :param save_model: Path to save the trained model (e.g., "model.keras").
        :type save_model: str   
        :param wandb_project: Name of the Weights & Biases project to log to.
        :type wandb_project: str
        """
        classifier = IntentClassifier(config=config, training_data=training_data, wandb_project=wandb_project)
        classifier.train(save_model=save_model)
        print("Training completed successfully!")

    def predict(load_model: str, input_text: str, wandb_project: str = None):
        """
        Make predictions using a trained model.

        :param load_model: Path to the saved Keras model file or W&B URL.
        :type load_model: str
        :param input_text: The input text string to classify.
        :type input_text: str
        :param wandb_project: Name of the Weights & Biases project to log to.
        :type wandb_project: str
        """
        classifier = IntentClassifier(load_model=load_model, wandb_project=wandb_project)
        predictions = classifier.predict(input_text)
        print(f"Predictions: {predictions}")

    def cross_validation(config: str, training_data: str, n_splits: int = 3, wandb_project: str = None):
        """
        Run cross-validation on the model.

        :param config: Path to the YAML configuration file.
        :type config: str
        :param training_data: Path to the YAML file with training examples.
        :type training_data: str
        :param n_splits: The number of folds to use.
        :type n_splits: int, optional
        :param wandb_project: Name of the Weights & Biases project to log to.
        :type wandb_project: str
        """
        classifier = IntentClassifier(config=config, training_data=training_data, wandb_project=wandb_project)
        results = classifier.cross_validation(n_splits=n_splits)
        print("Cross-validation completed successfully!")
        pprint(results)

    fire.Fire({
        'train': train,
        'predict': predict,
        'cross_validation': cross_validation
    }, serialize=False)