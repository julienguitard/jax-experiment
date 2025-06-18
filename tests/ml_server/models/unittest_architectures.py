import unittest
from sources.common.constants import EPS
from sources.ml_server.models_.architectures import (
    layer_metadata_generator,
    layers_metadata_generator,
    pen_layer_metadata_generator,
    pen_metadata_generator,
    loss_metadata_generator,
    optimizer_model_metadata_generator,
)


class TestArchitecture(unittest.TestCase):
    def setUp(self):
        # Generate mock functions
        self.generate_layer_metadata = layer_metadata_generator()
        self.generate_layers_metadata = layers_metadata_generator(
            "relu_feedfoward", self.generate_layer_metadata
        )
        self.generate_pen_layer_metadata = pen_layer_metadata_generator()
        self.generate_pen_metadata = pen_metadata_generator(
            self.generate_pen_layer_metadata
        )
        self.generate_loss_metadata = loss_metadata_generator()
        self.generate_optimizer_model_metadata = (
            optimizer_model_metadata_generator()
        )

        # Set up common data for testing
        self.architecture_type = "relu_feedfoward"
        self.layer_sizes = [3, 16, 13, 12, 5]
        self.penalization_type = "l1"
        self.penalization_l = 1e-3
        self.optimizer_model_type = "sgd_momentum"
        self.learning_rate = 1e-5
        self.lps = [1e-3, 1e-4]
        self.lds = [1e-3, 1e-4]
        self.betas = [0.8]
        self.b_momentum = 0.8
        self.b_batch_norm = 0.9

    def test_generate_layer_metadata(self):
        # Call the generate_layer_metadata function
        linear = self.generate_layer_metadata(
            "linear", self.layer_sizes[0], self.layer_sizes[1]
        )
        relu = self.generate_layer_metadata(
            "relu", self.layer_sizes[0], self.layer_sizes[1]
        )
        batch_norm = self.generate_layer_metadata(
            "batch_norm", self.layer_sizes[0], self.layer_sizes[1]
        )
        result = [linear, relu, batch_norm]

        # Define the expected output
        expected_output = [
            {"type": "linear", "n_inputs": 3, "n_outputs": 16},
            {"type": "relu", "n_inputs": 3, "n_outputs": 16},
            {"type": "batch_norm", "n_inputs": 3, "n_outputs": 3},
        ]

        # Assert that the result matches the expected output
        self.assertEqual(result, expected_output)

    def test_generates_layers_generator(self):
        # Call the architecture_generator function
        result = self.generate_layers_metadata(self.layer_sizes)

        # Define the expected output
        expected_output = [
            {"type": "batch_norm", "n_inputs": 3, "n_outputs": 3},
            {"type": "relu", "n_inputs": 3, "n_outputs": 16},
            {"type": "batch_norm", "n_inputs": 16, "n_outputs": 16},
            {"type": "relu", "n_inputs": 16, "n_outputs": 13},
            {"type": "batch_norm", "n_inputs": 13, "n_outputs": 13},
            {"type": "relu", "n_inputs": 13, "n_outputs": 12},
            {"type": "batch_norm", "n_inputs": 12, "n_outputs": 12},
            {"type": "linear", "n_inputs": 12, "n_outputs": 5},
        ]

        # Assert that the result matches the expected output
        self.assertEqual(result, expected_output)

    def test_generate_layer_pen_metadata(self):
        # Call the function
        r0 = self.generate_pen_layer_metadata(
            "batch_norm", self.penalization_type, self.penalization_l
        )
        r1 = self.generate_pen_layer_metadata(
            "linear", self.penalization_type, self.penalization_l
        )
        r2 = self.generate_pen_layer_metadata(
            "relu", self.penalization_type, self.penalization_l
        )
        r3 = self.generate_pen_layer_metadata(
            "batch_norm", "l2", self.penalization_l
        )
        r4 = self.generate_pen_layer_metadata(
            "linear", "l2", self.penalization_l
        )
        r5 = self.generate_pen_layer_metadata(
            "relu", "l2", self.penalization_l
        )

        result = [r0, r1, r2, r3, r4, r5]

        # Define the expected output
        eo0 = {"pen": {"type": "none", "l": 0.0, "power": 1.0}}
        eo1 = {"pen": {"type": "l1", "l": self.penalization_l, "power": 1.0}}
        eo2 = {"pen": {"type": "l1", "l": self.penalization_l, "power": 1.0}}
        eo3 = {"pen": {"type": "none", "l": 0.0, "power": 1.0}}
        eo4 = {"pen": {"type": "l2", "l": self.penalization_l, "power": 2.0}}
        eo5 = {"pen": {"type": "l2", "l": self.penalization_l, "power": 2.0}}

        expected_output = [eo0, eo1, eo2, eo3, eo4, eo5]

        # Assert that the result matches the expected output
        self.assertEqual(result, expected_output)

    def test_generate_pen_metadata(self):
        # Define the input
        layers_metadata = [
            {"type": "batch_norm", "n_inputs": 3, "n_outputs": 3},
            {"type": "relu", "n_inputs": 3, "n_outputs": 16},
            {"type": "batch_norm", "n_inputs": 16, "n_outputs": 16},
            {"type": "relu", "n_inputs": 16, "n_outputs": 13},
            {"type": "batch_norm", "n_inputs": 13, "n_outputs": 13},
            {"type": "relu", "n_inputs": 13, "n_outputs": 12},
            {"type": "batch_norm", "n_inputs": 12, "n_outputs": 12},
            {"type": "linear", "n_inputs": 12, "n_outputs": 5},
        ]

        # Call the function
        results = self.generate_pen_metadata(
            layers_metadata, self.penalization_type, self.penalization_l
        )

        # Define the expected output
        expected_output = [
            {
                "type": "batch_norm",
                "n_inputs": 3,
                "n_outputs": 3,
                "pen": {"type": "none", "l": 0.0, "power": 1.0},
            },
            {
                "type": "relu",
                "n_inputs": 3,
                "n_outputs": 16,
                "pen": {"type": "l1", "l": self.penalization_l, "power": 1.0},
            },
            {
                "type": "batch_norm",
                "n_inputs": 16,
                "n_outputs": 16,
                "pen": {"type": "none", "l": 0.0, "power": 1.0},
            },
            {
                "type": "relu",
                "n_inputs": 16,
                "n_outputs": 13,
                "pen": {"type": "l1", "l": self.penalization_l, "power": 1.0},
            },
            {
                "type": "batch_norm",
                "n_inputs": 13,
                "n_outputs": 13,
                "pen": {"type": "none", "l": 0.0, "power": 1.0},
            },
            {
                "type": "relu",
                "n_inputs": 13,
                "n_outputs": 12,
                "pen": {"type": "l1", "l": self.penalization_l, "power": 1.0},
            },
            {
                "type": "batch_norm",
                "n_inputs": 12,
                "n_outputs": 12,
                "pen": {"type": "none", "l": 0.0, "power": 1.0},
            },
            {
                "type": "linear",
                "n_inputs": 12,
                "n_outputs": 5,
                "pen": {"type": "l1", "l": self.penalization_l, "power": 1.0},
            },
        ]

        # Assert that the result matches the expected output
        self.assertEqual(results, expected_output)

    def test_generate_loss_metadata(self):
        # Define the input
        layers_metadata = [
            {
                "type": "batch_norm",
                "n_inputs": 3,
                "n_outputs": 3,
                "pen": {"type": "none", "l": 0.0, "power": 1.0},
            },
            {
                "type": "relu",
                "n_inputs": 3,
                "n_outputs": 16,
                "pen": {"type": "l1", "l": self.penalization_l, "power": 1.0},
            },
            {
                "type": "batch_norm",
                "n_inputs": 16,
                "n_outputs": 16,
                "pen": {"type": "none", "l": 0.0, "power": 1.0},
            },
            {
                "type": "relu",
                "n_inputs": 16,
                "n_outputs": 13,
                "pen": {"type": "l1", "l": self.penalization_l, "power": 1.0},
            },
            {
                "type": "batch_norm",
                "n_inputs": 13,
                "n_outputs": 13,
                "pen": {"type": "none", "l": 0.0, "power": 1.0},
            },
            {
                "type": "relu",
                "n_inputs": 13,
                "n_outputs": 12,
                "pen": {"type": "l1", "l": self.penalization_l, "power": 1.0},
            },
            {
                "type": "batch_norm",
                "n_inputs": 12,
                "n_outputs": 12,
                "pen": {"type": "none", "l": 0.0, "power": 1.0},
            },
            {
                "type": "linear",
                "n_inputs": 12,
                "n_outputs": 5,
                "pen": {"type": "l1", "l": self.penalization_l, "power": 1.0},
            },
        ]

        # Call the function
        results = self.generate_loss_metadata(layers_metadata, "mse")

        # Define the expected output
        expected_output = {
            "loss_type": "mse",
            "layers": [
                {
                    "type": "batch_norm",
                    "n_inputs": 3,
                    "n_outputs": 3,
                    "pen": {"type": "none", "l": 0.0, "power": 1.0},
                },
                {
                    "type": "relu",
                    "n_inputs": 3,
                    "n_outputs": 16,
                    "pen": {
                        "type": "l1",
                        "l": self.penalization_l,
                        "power": 1.0,
                    },
                },
                {
                    "type": "batch_norm",
                    "n_inputs": 16,
                    "n_outputs": 16,
                    "pen": {"type": "none", "l": 0.0, "power": 1.0},
                },
                {
                    "type": "relu",
                    "n_inputs": 16,
                    "n_outputs": 13,
                    "pen": {
                        "type": "l1",
                        "l": self.penalization_l,
                        "power": 1.0,
                    },
                },
                {
                    "type": "batch_norm",
                    "n_inputs": 13,
                    "n_outputs": 13,
                    "pen": {"type": "none", "l": 0.0, "power": 1.0},
                },
                {
                    "type": "relu",
                    "n_inputs": 13,
                    "n_outputs": 12,
                    "pen": {
                        "type": "l1",
                        "l": self.penalization_l,
                        "power": 1.0,
                    },
                },
                {
                    "type": "batch_norm",
                    "n_inputs": 12,
                    "n_outputs": 12,
                    "pen": {"type": "none", "l": 0.0, "power": 1.0},
                },
                {
                    "type": "linear",
                    "n_inputs": 12,
                    "n_outputs": 5,
                    "pen": {
                        "type": "l1",
                        "l": self.penalization_l,
                        "power": 1.0,
                    },
                },
            ],
        }

        # Assert that the result matches the expected output
        self.assertEqual(results, expected_output)

    def test_generate_optimizer_model_metadata(self):
        # Set up test data
        archi = {
            "loss_type": "mse",
            "layers": [
                {
                    "type": "batch_norm",
                    "n_inputs": 3,
                    "n_outputs": 3,
                    "pen": {"type": "none", "l": 0.0, "power": 1.0},
                },
                {
                    "type": "relu",
                    "n_inputs": 3,
                    "n_outputs": 16,
                    "pen": {
                        "type": "l1",
                        "l": self.penalization_l,
                        "power": 1.0,
                    },
                },
                {
                    "type": "batch_norm",
                    "n_inputs": 16,
                    "n_outputs": 16,
                    "pen": {"type": "none", "l": 0.0, "power": 1.0},
                },
                {
                    "type": "relu",
                    "n_inputs": 16,
                    "n_outputs": 13,
                    "pen": {
                        "type": "l1",
                        "l": self.penalization_l,
                        "power": 1.0,
                    },
                },
                {
                    "type": "batch_norm",
                    "n_inputs": 13,
                    "n_outputs": 13,
                    "pen": {"type": "none", "l": 0.0, "power": 1.0},
                },
                {
                    "type": "relu",
                    "n_inputs": 13,
                    "n_outputs": 12,
                    "pen": {
                        "type": "l1",
                        "l": self.penalization_l,
                        "power": 1.0,
                    },
                },
                {
                    "type": "batch_norm",
                    "n_inputs": 12,
                    "n_outputs": 12,
                    "pen": {"type": "none", "l": 0.0, "power": 1.0},
                },
                {
                    "type": "linear",
                    "n_inputs": 12,
                    "n_outputs": 5,
                    "pen": {
                        "type": "l1",
                        "l": self.penalization_l,
                        "power": 1.0,
                    },
                },
            ],
        }

        penalization_type = "l2"
        penalization_l = 0.1
        model_params = {"weights": [1.0, 2.0], "biases": [0.5, 0.5]}

        # Call the generate_optimizer_model_metadata function
        result = self.generate_optimizer_model_metadata(
            archi,
            self.optimizer_model_type,
            self.learning_rate,
            self.lps,
            self.lds,
            betas=self.betas,
        )

        # Define the expected output
        expected_output = {
            "loss_type": "mse",
            "layers": [
                {
                    "type": "batch_norm",
                    "n_inputs": 3,
                    "n_outputs": 3,
                    "pen": {"type": "none", "l": 0.0, "power": 1.0},
                },
                {
                    "type": "relu",
                    "n_inputs": 3,
                    "n_outputs": 16,
                    "pen": {
                        "type": "l1",
                        "l": self.penalization_l,
                        "power": 1.0,
                    },
                },
                {
                    "type": "batch_norm",
                    "n_inputs": 16,
                    "n_outputs": 16,
                    "pen": {"type": "none", "l": 0.0, "power": 1.0},
                },
                {
                    "type": "relu",
                    "n_inputs": 16,
                    "n_outputs": 13,
                    "pen": {
                        "type": "l1",
                        "l": self.penalization_l,
                        "power": 1.0,
                    },
                },
                {
                    "type": "batch_norm",
                    "n_inputs": 13,
                    "n_outputs": 13,
                    "pen": {"type": "none", "l": 0.0, "power": 1.0},
                },
                {
                    "type": "relu",
                    "n_inputs": 13,
                    "n_outputs": 12,
                    "pen": {
                        "type": "l1",
                        "l": self.penalization_l,
                        "power": 1.0,
                    },
                },
                {
                    "type": "batch_norm",
                    "n_inputs": 12,
                    "n_outputs": 12,
                    "pen": {"type": "none", "l": 0.0, "power": 1.0},
                },
                {
                    "type": "linear",
                    "n_inputs": 12,
                    "n_outputs": 5,
                    "pen": {
                        "type": "l1",
                        "l": self.penalization_l,
                        "power": 1.0,
                    },
                },
            ],
            "optimizer_params": {
                "type": "sgd_momentum",
                "lr": self.learning_rate,
                "lps": self.lps,
                "lds": self.lds,
                "betas": self.betas,
                "b_batch_norm": self.b_batch_norm,
                "eps": EPS,
            },
        }

        # Assert that the result matches the expected output
        self.assertEqual(result, expected_output)


if __name__ == "__main__":
    unittest.main()
