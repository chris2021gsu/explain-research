"""explainkit: plain-language, tool-grounded explanations for regression and forecasting models."""

from explainkit.explainer import Explainer
from explainkit.grounding import unsupported_numbers
from explainkit.llm import LiteLLM
from explainkit.narrate import Result
from explainkit.report import Answer, Report
from explainkit.tools import Explanation, explain_change, explain_prediction, what_if

__all__ = ["Answer", "Explainer", "Explanation", "LiteLLM", "Report", "Result", "explain_change", "explain_prediction",
           "unsupported_numbers", "what_if"]
__version__ = "0.1.0"
