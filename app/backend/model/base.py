from abc import ABC, abstractmethod


class PrefOptimBase(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.status = "train"

    @abstractmethod
    def compute_utility_raw(self, z):
        """
        Compute the utility of z without considering the scaling issue.
        """
        pass

    def compute_utility(self, z):
        if self.status == "train":
            print("computing training utility")
            return self.compute_utility_raw(z)

        elif self.status == "test":
            print("computing test utility")
            return self.compute_utility_inverse_normalized(z)

        else:
            raise ValueError(f"Unimplemented status: {self.status}")

    def compute_utility_inverse_normalized(self, z):
        """
        Assuming that the model's saved y have already been normalized, (so that compute_utility_raw is also normalized),
        compute the un-normalized prediction.
        """

        norm_u = self.compute_utility_raw(z)
        if self.y_std == 0:
            return norm_u
        else:
            return norm_u * self.y_std + self.y_mean

    def ends_training(self):
        print("Model training is completed, switched to test mode")
        self.status = "test"

        self.y_mean = self.y.mean()
        self.y_std = self.y.std()

        if self.y_std != 0:
            self.y = (self.y - self.y_mean) / self.y_std
