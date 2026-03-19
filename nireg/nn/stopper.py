class EarlyStopper:
    """
    A utility class for implementing an early stopping mechanism for training models.

    Early stopping is used to halt the training process when a monitored quantity
    (e.g., validation loss) stops improving to avoid overfitting or unnecessary computations.

    Attributes
    ----------
    patience : int
        The number of consecutive epochs that the monitored quantity is allowed to stop
        improving before stopping the training process.
    min_delta : float
        The minimum change in the monitored quantity to qualify as an improvement.
    counter : int
        Counts the number of epochs where the monitored quantity has not improved beyond
        the `min_delta` threshold.
    min_validation_loss : float
        The lowest observed value of the monitored quantity during training.

    Methods
    -------
    early_stop_old(validation_loss: float) -> bool:
        Deprecated: Checks if the training should be stopped based on patience and delta criteria.
    early_stop(validation_loss: float) -> bool:
        Modern implementation that tracks loss changes and determines if training should halt.
    """

    def __init__(self, patience: int = 1, min_delta: float = 0):
        """
        Initialize an EarlyStopper instance.

        Parameters
        ----------
        patience : int, optional
            The number of consecutive epochs to wait for an improvement in the monitored
            quantity before stopping the training, by default 1.
        min_delta : float, optional
            The minimum change in the monitored quantity to qualify as an improvement, by default 0.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop_old(self, validation_loss: float) -> bool:
        """
        Deprecated: Check whether training should stop based on old early stopping logic.

        This function considers a monitored quantity to be improving only if it decreases
        compared to the previous best by an amount greater than `min_delta`.

        Parameters
        ----------
        validation_loss : float
            The current validation loss being monitored.

        Returns
        -------
        bool
            True if the training should stop, False otherwise.
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def early_stop(self, validation_loss: float) -> bool:
        """
        Check whether training should stop based on improved early stopping logic.

        This method assesses whether there has been no significant change in the monitored
        quantity for the amount of time specified by `patience`. It is robust against small
        oscillations in the loss values within the range defined by `min_delta`.

        Parameters
        ----------
        validation_loss : float
            The current validation loss being monitored.

        Returns
        -------
        bool
            True if the training should stop, False otherwise.
        """
        if abs(validation_loss - self.min_validation_loss) < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.counter = 0
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
        return False
