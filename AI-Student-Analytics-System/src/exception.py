"""
src/exception.py
================
Custom exception with full traceback details.
Beginners: This makes error messages much easier to understand
           by showing EXACTLY which file and line caused the error.
"""
import sys


def get_error_message(error, error_detail: sys):
    """Build a detailed error string with file name and line number."""
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_no   = exc_tb.tb_lineno
    return (
        f"Error in script: [{file_name}]  "
        f"at line [{line_no}]  "
        f">> {str(error)}"
    )


class StudentAnalyticsException(Exception):
    """
    Custom exception for the project.
    Usage:
        try:
            ...
        except Exception as e:
            raise StudentAnalyticsException(e, sys)
    """
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = get_error_message(
            error_message, error_detail=error_detail
        )

    def __str__(self):
        return self.error_message
