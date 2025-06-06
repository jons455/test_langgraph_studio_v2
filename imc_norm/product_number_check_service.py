from abc import ABC, abstractmethod
from typing import List, Dict, Any

class ProductNumberCheckService(ABC):
    @abstractmethod
    def validate_product_number(self, product_number: str) -> Dict[str, Any]:
        """
        Validate a single product number (MLFB) and return the validation result.

        :param product_number: The product number (MLFB) to validate.
        :return: A dictionary containing the validation results.
        """
        pass

    @abstractmethod
    def validate_product_numbers_batch(self, product_numbers: List[str]) -> List[Dict[str, Any]]:
        """
        Validate a batch of product numbers (MLFBs) and return the validation results.

        :param product_numbers: A list of product numbers (MLFBs) to validate.
        :return: A list of dictionaries, each containing the validation result per product number.
        """
        pass
