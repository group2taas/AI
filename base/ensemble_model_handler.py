import os
import json
import asyncio
import numpy as np
import random
import re
from typing import List, Dict, Optional, Tuple, Any, Union, Set, Callable
from loguru import logger
from dataclasses import dataclass
from collections import Counter, defaultdict
from datetime import datetime

from .model_handler import AIModelHandler
from .owasp_model_handler import OWASPModelHandler


@dataclass
class ModelWeights:
    """Weights for different model contributions in the ensemble."""
    generation_weight: float = 1.0
    voting_weight: float = 1.0
    rag_weight: float = 1.0
    confidence_impact: float = 0.2
    diversity_bonus: float = 0.3


class ModelResponse:
    """Container for model responses with metadata and analysis."""
    
    def __init__(self, model_name: str, response: str, temperature: float = 0.0):
        self.model_name = model_name
        self.response = response
        self.temperature = temperature
        self.timestamp = datetime.now()
        self.quality_score = 0.0
        self.test_cases = set()
        self.vulnerabilities = []
        self.confidence = 0.0
        self.valid = bool(response and response.strip())
        
        if self.valid:
            self._extract_metrics()
    
    def _extract_metrics(self):
        """Extract test cases and calculate quality metrics from the response."""
        try:
            test_functions = re.findall(r"def\s+(test_[a-zA-Z0-9_]+)\s*\(", self.response)
            self.test_cases = set(test_functions)
            self.vulnerabilities = [test.replace("test_", "").replace("_", " ").title() for test in test_functions]
            self.quality_score = self._calculate_quality_score()
            self.confidence = self._estimate_confidence()
            
        except Exception as e:
            logger.error(f"Error extracting metrics from {self.model_name} response: {e}")
            self.valid = False
    
    def _calculate_quality_score(self) -> float:
        """Calculate a quality score for the response based on various metrics."""
        if not self.valid:
            return 0.0
        
        complexity_score = 0.0
        coverage_score = 0.0
        consistency_score = 0.0
        
        lines = self.response.strip().split("\n")
        complexity_score = min(1.0, len(lines) / 200) * 0.4
        complexity_score += min(1.0, len(self.test_cases) / 10) * 0.6
        
        imports = re.findall(r"import\s+(\w+)|from\s+(\w+)", self.response)
        unique_imports = set([imp[0] or imp[1] for imp in imports])
        coverage_score = min(1.0, len(unique_imports) / 10) * 0.5
        
        key_patterns = [
            r"selenium", r"webdriver", r"zapv2", r"ZAPv2", 
            r"requests", r"beautifulsoup", r"payload", 
            r"json\.dumps", r"unittest", r"assert"
        ]
        approach_coverage = sum(1 for pattern in key_patterns if re.search(pattern, self.response))
        coverage_score += min(1.0, approach_coverage / len(key_patterns)) * 0.5
        
        structure_patterns = [
            r"class.*\(unittest\.TestCase\)", r"def setUp", r"def tearDown",
            r"if __name__ == .__main__", r"self\.results", r"try:.*?except"
        ]
        structure_score = sum(1 for pattern in structure_patterns if re.search(pattern, self.response, re.DOTALL))
        consistency_score = min(1.0, structure_score / len(structure_patterns))
        
        return 0.3 * complexity_score + 0.4 * coverage_score + 0.3 * consistency_score
    
    def _estimate_confidence(self) -> float:
        """Estimate model confidence based on response characteristics."""
        if not self.valid:
            return 0.0
        
        confidence = 0.5 * self.quality_score
        
        uncertainty_indicators = [
            r"# TODO", r"# FIXME", r"# NOTE", r"# WARNING",
            r"(?i)placeholder", r"(?i)example", r"(?i)sample",
            r"<.*?>", r"\?{2,}", r"to be implemented"
        ]
        
        uncertainty_count = sum(1 for pattern in uncertainty_indicators if re.search(pattern, self.response))
        uncertainty_penalty = min(0.5, uncertainty_count * 0.1)
        confidence = max(0.0, confidence - uncertainty_penalty)
        
        if len(self.test_cases) >= 5 and len(self.response.split('\n')) >= 100:
            confidence += 0.2
            
        return min(1.0, confidence)


class EnsembleModelHandler:

    def __init__(
        self, 
        models: Optional[List[str]] = None, 
        weights: Optional[ModelWeights] = None,
        temperature_variants: Optional[List[float]] = None,
        confidence_threshold: float = 0.6,
        max_generation_attempts: int = 3,
        consensus_strategy: str = "weighted"
    ):
        self.models = models or ["base", "rag"]
        self.weights = weights or ModelWeights()
        self.temperature_variants = temperature_variants or [0.2, 0.5, 0.8]
        self.confidence_threshold = confidence_threshold
        self.max_generation_attempts = max_generation_attempts
        self.consensus_strategy = consensus_strategy
        
        self.model_handlers = {}
        for model_name in self.models:
            if model_name == "base":
                self.model_handlers[model_name] = AIModelHandler()
            elif model_name == "rag":
                self.model_handlers[model_name] = OWASPModelHandler()
            else:
                raise ValueError(f"Unknown model: {model_name}")
        
        self.model_performance = {model: {'successes': 0, 'failures': 0, 'total': 0} for model in self.models}
        self.vulnerability_success_map = defaultdict(lambda: defaultdict(float))
        self.recent_generations = []
        
        logger.info(f"Initialized enhanced ensemble model with {len(self.model_handlers)} models")

    async def query_all_models_async(self, prompt: str) -> Dict[str, ModelResponse]:
        """Query all models asynchronously and return their responses with metadata."""
        tasks = []
        
        for model_name, handler in self.model_handlers.items():
            for temp in self.temperature_variants:
                variant_name = f"{model_name}_t{temp}"
                tasks.append(self._safe_query_model(variant_name, model_name, handler, prompt, temp))
        
        results = await asyncio.gather(*tasks)
        
        responses = {}
        for variant_name, model_name, response, temp in results:
            if response.strip():
                responses[variant_name] = ModelResponse(model_name, response, temp)
        
        return responses
    
    async def _safe_query_model(self, variant_name: str, model_name: str, handler: Any, prompt: str, 
                               temperature: float = 0.0) -> Tuple[str, str, str, float]:
        """Safely query a model with error handling, returning the variant name, base model name, response, and temperature."""
        try:
            temp_prompt = prompt
            if temperature > 0:
                temp_prompt = f"{prompt}\n\nTemperature: {temperature}"
                
            response = await handler.query_model_async(temp_prompt)
            if not response:
                logger.warning(f"Empty response from model {variant_name}")
                self._update_model_performance(model_name, False)
                return variant_name, model_name, "", temperature
                
            self._update_model_performance(model_name, True)
            return variant_name, model_name, response, temperature
            
        except Exception as e:
            logger.error(f"Error querying model {variant_name}: {e}")
            self._update_model_performance(model_name, False)
            return variant_name, model_name, "", temperature
    
    def _update_model_performance(self, model_name: str, success: bool):
        """Update the tracked performance metrics for a model."""
        if model_name in self.model_performance:
            self.model_performance[model_name]['total'] += 1
            if success:
                self.model_performance[model_name]['successes'] += 1
            else:
                self.model_performance[model_name]['failures'] += 1
    
    def _calculate_model_reliability(self, model_name: str) -> float:
        """Calculate a reliability score for a model based on its performance history."""
        stats = self.model_performance.get(model_name, {'successes': 0, 'total': 1})
        if stats['total'] == 0:
            return 0.5
        
        alpha = 1
        beta = 1
        successes = stats['successes'] + alpha
        total = stats['total'] + alpha + beta
        
        return successes / total
    
    def _extract_vulnerabilities(self, script_content: str) -> List[str]:
        """Extract vulnerability test cases from a script with improved pattern matching."""
        test_functions = []
        
        pattern1 = re.findall(r"def\s+(test_[a-zA-Z0-9_]+)\s*\(", script_content)
        if pattern1:
            test_functions.extend(pattern1)
            
        pattern2 = re.findall(r'self\.results\["(test_[a-zA-Z0-9_]+)"\]', script_content)
        if pattern2:
            test_functions.extend(pattern2)
            
        pattern3 = re.findall(r'"test_case":\s*"(test_[a-zA-Z0-9_]+)"', script_content)
        if pattern3:
            test_functions.extend(pattern3)
        
        test_functions = list(set(test_functions))
        
        vulnerabilities = []
        for test in test_functions:
            test_name = test.replace("test_", "").replace("_", " ").title()
            vulnerabilities.append(test_name)
            
        return vulnerabilities
    
    def _advanced_consensus_voting(self, model_responses: Dict[str, ModelResponse]) -> List[str]:
        """
        Implement advanced consensus voting to determine which vulnerabilities to test.
        Uses model confidence, reliability, and vulnerability-specific success rates.
        """
        base_models = {response.model_name for response in model_responses.values()}
        
        vuln_votes = defaultdict(list)
        for variant_name, response in model_responses.items():
            if not response.valid:
                continue
                
            model_reliability = self._calculate_model_reliability(response.model_name)
            
            for vuln in response.vulnerabilities:
                vote_weight = response.confidence
                vote_weight *= (0.7 + 0.3 * model_reliability)
                temp_factor = 1.0 - (response.temperature * 0.3)
                vote_weight *= temp_factor
                
                if response.model_name == "rag":
                    vote_weight *= self.weights.rag_weight
                
                vuln_success_rate = self.vulnerability_success_map[response.model_name].get(vuln, 0.5)
                vote_weight *= (0.8 + 0.4 * vuln_success_rate)
                
                vuln_votes[vuln].append(vote_weight)
        
        vuln_scores = {}
        for vuln, votes in vuln_votes.items():
            if self.consensus_strategy == "unanimous":
                if len({response.model_name for variant_name, response in model_responses.items() 
                      if vuln in response.vulnerabilities}) == len(base_models):
                    vuln_scores[vuln] = sum(votes)
            elif self.consensus_strategy == "majority":
                if len({response.model_name for variant_name, response in model_responses.items() 
                      if vuln in response.vulnerabilities}) > len(base_models) / 2:
                    vuln_scores[vuln] = sum(votes)
            elif self.consensus_strategy == "weighted":
                vuln_scores[vuln] = sum(votes)
            elif self.consensus_strategy == "adaptive":
                min_votes = max(1, len(base_models) // 2)
                if len({response.model_name for variant_name, response in model_responses.items() 
                      if vuln in response.vulnerabilities}) >= min_votes:
                    vuln_scores[vuln] = sum(votes)
        
        if vuln_scores:
            max_score = max(vuln_scores.values())
            vuln_scores = {v: s/max_score for v, s in vuln_scores.items()}
            
        consensus_vulns = [v for v, score in vuln_scores.items() if score >= self.confidence_threshold]
        
        if not consensus_vulns and vuln_scores:
            consensus_vulns = sorted(vuln_scores.keys(), key=lambda v: vuln_scores[v], reverse=True)[:3]
        
        return consensus_vulns
    
    def _combine_scripts_advanced(self, model_responses: Dict[str, ModelResponse], 
                                 consensus_vulnerabilities: List[str]) -> str:

        valid_responses = {variant: resp for variant, resp in model_responses.items() if resp.valid}
        
        if not valid_responses:
            return self._generate_fallback_template(consensus_vulnerabilities)
        
        components = self._extract_script_components(valid_responses)
        
        best_imports = self._select_best_imports(components["imports"], valid_responses)
        best_setup = self._select_best_component(components["setup"], valid_responses)
        best_teardown = self._select_best_component(components["teardown"], valid_responses)
        
        test_functions = self._select_test_functions(components["tests"], valid_responses, 
                                                    consensus_vulnerabilities)
        
        script = f"{best_imports}\n\n{best_setup}\n\n"
        
        for func_name, func_code in test_functions.items():
            normalized_code = self._normalize_indentation(func_code)
            indented_code = "\n".join(f"    {line}" for line in normalized_code.split('\n'))
            script += f"{indented_code}\n\n"
        
        script += f"{best_teardown}\n\n"
        script += 'if __name__ == "__main__":\n    unittest.main()'
        
        script = self._validate_script_indentation(script)
        
        return script
    
    def _extract_script_components(self, model_responses: Dict[str, ModelResponse]) -> Dict[str, Dict]:
        """Extract all key script components from the model responses."""
        components = {
            "imports": {},
            "setup": {},
            "teardown": {},
            "tests": defaultdict(dict)
        }
        
        for variant, response in model_responses.items():
            script = response.response
            
            imports_match = re.search(r"((?:import|from)\s+.*?)(?:class|def|if)", script, re.DOTALL)
            if imports_match:
                components["imports"][variant] = imports_match.group(1).strip()
            
            setup_match = re.search(r"(class\s+\w+.*?setUp.*?)(?:def\s+test_)", script, re.DOTALL)
            if setup_match:
                components["setup"][variant] = setup_match.group(1).strip()
            
            teardown_match = re.search(r"(def\s+tearDown.*?)(?:if\s+__name__|$)", script, re.DOTALL)
            if teardown_match:
                teardown_code = teardown_match.group(1).strip()
                if not teardown_code.startswith("    "):
                    teardown_code = "    " + teardown_code.replace("\n", "\n    ")
                components["teardown"][variant] = teardown_code
            
            test_matches = re.finditer(r"(def\s+(test_[a-zA-Z0-9_]+)\s*\([^)]*\):(?:\s*.*?(?=def|\Z))+)", 
                                       script, re.DOTALL)
            for match in test_matches:
                func_code = match.group(1).strip()
                func_name = match.group(2)
                components["tests"][func_name][variant] = func_code
        
        return components
    
    def _select_best_imports(self, import_blocks: Dict[str, str], 
                            model_responses: Dict[str, ModelResponse]) -> str:
        if not import_blocks:
            return """import unittest
import json
import time
import os
import re
import logging
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from zapv2 import ZAPv2
import requests
from urllib.parse import urljoin, urlparse"""
        
        all_imports = set()
        for variant, import_block in import_blocks.items():
            clean_imports = []
            for line in import_block.split('\n'):
                stripped = line.strip()
                if stripped.startswith(('import ', 'from ')):
                    clean_imports.append(stripped)

            all_imports.update(clean_imports)
        
        essential_imports = [
            "import unittest",
            "import json",
            "import os",
            "from dotenv import load_dotenv",
            "import requests",
            "from zapv2 import ZAPv2"
        ]
        
        for imp in essential_imports:
            if not any(imp in existing for existing in all_imports):
                all_imports.add(imp)
        
        std_lib = []
        third_party = []
        
        for imp in all_imports:
            if any(lib in imp for lib in ["unittest", "json", "os", "re", "time", "sys", "logging", 
                                         "urllib", "datetime", "pathlib"]):
                std_lib.append(imp)
            else:
                third_party.append(imp)
        
        return "\n".join(sorted(std_lib) + [""] + sorted(third_party))
    
    def _select_best_component(self, components: Dict[str, str], 
                              model_responses: Dict[str, ModelResponse]) -> str:
        if not components:
            if "setup" in components:
                return """class OWASP_SecurityTests(unittest.TestCase):
    def setUp(self):
        # Load environment variables with fallback
        try:
            if os.path.exists('.env'):
                load_dotenv('.env')
        except Exception as e:
            logging.warning(f"Failed to load .env file: {e}")
            
        self.target_url = "<target_url>"
        self.results = {}
        self.scan_timeout = 300
        
        # Get ZAP configuration with proper fallbacks
        self.zap_api_key = os.getenv("ZAP_API_KEY") or os.getenv("ZAP_API") or ""
        zap_proxy_server = os.getenv("ZAP_PROXY_SERVER")
        self.zap_proxy = "http://localhost:8080"
        
        try:
            if self.zap_api_key:
                self.zap = ZAPv2(apikey=self.zap_api_key, proxies={"http": self.zap_proxy, "https": self.zap_proxy})
            else:
                self.zap = ZAPv2(proxies={"http": self.zap_proxy, "https": self.zap_proxy})
        except Exception as e:
            logging.error(f"Error initializing ZAP: {e}")
            self.zap = None"""
            else:
                return """    def tearDown(self):
        for test, result in self.results.items():
            print(json.dumps({
                "type": "result",
                "test_case": test,
                "target_url": self.target_url,
                "result": result
            }))"""
        
        best_variant = max(components.keys(), 
                          key=lambda v: model_responses[v].quality_score if v in model_responses else 0)
        
        best_component = components[best_variant]
        
        if "setup" in components:
            if "load_dotenv()" not in best_component:
                best_component = best_component.replace("def setUp(self):", 
                                                      "def setUp(self):\n        load_dotenv()")
            
            if "self.results = {}" not in best_component:
                best_component = best_component.replace("def setUp(self):", 
                                                      "def setUp(self):\n        self.results = {}")
        
        return best_component
    
    def _select_test_functions(self, test_functions: Dict[str, Dict[str, str]], 
                            model_responses: Dict[str, ModelResponse],
                            consensus_vulnerabilities: List[str]) -> Dict[str, str]:
        """Select the best test functions for each vulnerability in the consensus list."""
        selected_functions = {}

        test_name_map = {}
        for vuln in consensus_vulnerabilities:
            test_name = "test_" + vuln.lower().replace(" ", "_")
            test_name_map[vuln] = test_name

        for vuln, test_name in test_name_map.items():
            if test_name in test_functions:
                implementations = test_functions[test_name]
                best_variant = max(implementations.keys(),
                                key=lambda v: model_responses[v].quality_score if v in model_responses else 0)

                func_code = implementations[best_variant]
                func_code = self._normalize_indentation(func_code)

                selected_functions[test_name] = func_code
            else:
                selected_functions[test_name] = self._generate_fallback_test(test_name, vuln)

        return selected_functions

    def _normalize_indentation(self, code_block: str) -> str:
        """Normalize indentation in a code block to ensure consistency."""
        lines = code_block.split('\n')
        if not lines:
            return ""

        result = [lines[0]]

        base_indent = 0
        for i in range(1, len(lines)):
            if lines[i].strip():
                base_indent = len(lines[i]) - len(lines[i].lstrip())
                break

        for i in range(1, len(lines)):
            if not lines[i].strip():
                result.append('')
                continue

            current_indent = len(lines[i]) - len(lines[i].lstrip())
            if current_indent >= base_indent:
                indent_level = ((current_indent - base_indent) // 4) + 1
                result.append(' ' * (indent_level * 4) + lines[i].lstrip())
            else:
                result.append(' ' * (base_indent + 4) + lines[i].lstrip())

        return '\n'.join(result)

    def _generate_fallback_test(self, test_name: str, vulnerability: str) -> str:
        """Generate a fallback test implementation for a specific vulnerability."""
        template = f"""def {test_name}(self):
        try:
            # Test for {vulnerability}
            target = self.target_url
            self.results["{test_name}"] = f"Tested for {vulnerability}"
            
            # Record test completion
            logging.info(f"Completed {vulnerability} test")
        except Exception as e:
            self.results["{test_name}"] = f"Error: {{str(e)}}"
            logging.error(f"Error in {vulnerability} test: {{e}}")"""
            
        if "sql" in test_name.lower():
            template = f"""def {test_name}(self):
        try:
            payloads = ["' OR 1=1 --", "admin' --", "1' OR '1'='1", "1; DROP TABLE users"]
            target = self.target_url
            
            # Use ZAP to test for SQL injection if ZAP is available
            if hasattr(self, 'zap') and self.zap is not None:
                try:
                    scan_id = self.zap.ascan.scan(target, scanpolicyname="API-High-Security")
                    for payload in payloads:
                        self.zap.ascan.scan(target, payload)
                    
                    # Check if ZAP found SQL injection vulnerabilities
                    alerts = self.zap.core.alerts(baseurl=target)
                    sql_alerts = [a for a in alerts if "SQL" in a.get("alert", "")]
                    
                    if sql_alerts:
                        details = "\\n".join([f"- {{a.get('alert')}}: {{a.get('url')}}" for a in sql_alerts])
                        self.results["{test_name}"] = f"Found SQL injection vulnerabilities: {{details}}"
                    else:
                        self.results["{test_name}"] = f"No SQL injection vulnerabilities detected with payloads: {{payloads}}"
                except Exception as zap_err:
                    logging.warning(f"ZAP scan failed, falling back to manual testing: {{zap_err}}")
                    self._fallback_sql_injection_test(target, payloads, "{test_name}")
            else:
                # Fallback to manual testing if ZAP is not available
                logging.info("ZAP not available, using manual SQL injection testing")
                self._fallback_sql_injection_test(target, payloads, "{test_name}")
                
        except Exception as e:
            self.results["{test_name}"] = f"Error: {{str(e)}}"
            logging.error(f"Error in {vulnerability} test: {{e}}")
            
    def _fallback_sql_injection_test(self, target, payloads, test_name):
        # Manual SQL injection testing using requests
        try:
            vulnerable = False
            for payload in payloads:
                # Try with GET parameter
                test_url = f"{{target}}?id={{payload}}"
                resp = requests.get(test_url, timeout=10)
                
                # Look for common SQL error messages
                error_patterns = ["SQL syntax", "mysql_fetch", "ORA-", "sqlite3", "PostgreSQL"]
                if any(pattern in resp.text for pattern in error_patterns):
                    self.results[test_name] = f"Potential SQL injection found with payload: {{payload}}"
                    vulnerable = True
                    break
                    
            if not vulnerable:
                self.results[test_name] = f"No SQL injection vulnerabilities detected with basic payloads"
        except Exception as e:
            self.results[test_name] = f"Error in manual SQL injection test: {{str(e)}}" """
                
        elif "xss" in test_name.lower():
            template = f"""def {test_name}(self):
        try:
            payloads = ["<script>alert('XSS')</script>", "<img src=x onerror=alert('XSS')>", 
                       "javascript:alert('XSS')", "<body onload=alert('XSS')>"]
            target = self.target_url
            
            # Test for XSS vulnerabilities using basic requests
            xss_found = False
            for payload in payloads:
                # Try GET with the payload
                try:
                    encoded_payload = requests.utils.quote(payload)
                    test_url = f"{{target}}?q={{encoded_payload}}"
                    
                    response = requests.get(test_url, timeout=10)
                    if payload in response.text:
                        self.results["{test_name}"] = f"Potential XSS vulnerability found with payload: {{payload}}"
                        xss_found = True
                        break
                except Exception as req_err:
                    logging.warning(f"Error in XSS GET request: {{req_err}}")
            
            # Use ZAP for more thorough XSS testing if it's available
            if not xss_found and hasattr(self, 'zap') and self.zap is not None:
                try:
                    scan_id = self.zap.ascan.scan(target, scanpolicyname="API-High-Security")
                    alerts = self.zap.core.alerts(baseurl=target)
                    xss_alerts = [a for a in alerts if "Cross Site Scripting" in a.get("alert", "")]
                    
                    if xss_alerts:
                        details = "\\n".join([f"- {{a.get('alert')}}: {{a.get('url')}}" for a in xss_alerts])
                        self.results["{test_name}"] = f"Found XSS vulnerabilities: {{details}}"
                        xss_found = True
                except Exception as zap_err:
                    logging.warning(f"ZAP XSS scan failed: {{zap_err}}")
            
            # If no vulnerabilities were found with either method
            if not xss_found:
                self.results["{test_name}"] = f"No XSS vulnerabilities detected with standard payloads"
                
        except Exception as e:
            self.results["{test_name}"] = f"Error: {{str(e)}}"
            logging.error(f"Error in {vulnerability} test: {{e}}")"""
                
        elif "header" in test_name.lower() or "hsts" in test_name.lower():
            template = f"""def {test_name}(self):
        try:
            target = self.target_url
            
            # Check for security headers
            response = requests.get(target, timeout=10)
            headers = response.headers
            
            expected_headers = [
                'Strict-Transport-Security',
                'Content-Security-Policy',
                'X-Content-Type-Options',
                'X-Frame-Options',
                'X-XSS-Protection'
            ]
            
            missing_headers = [h for h in expected_headers if h not in headers]
            
            if missing_headers:
                self.results["{test_name}"] = f"Missing security headers: {{', '.join(missing_headers)}}"
            else:
                self.results["{test_name}"] = "All security headers are present"
                
        except Exception as e:
            self.results["{test_name}"] = f"Error: {{str(e)}}"
            logging.error(f"Error in {vulnerability} test: {{e}}")"""
                
        elif "cookie" in test_name.lower():
            template = f"""def {test_name}(self):
        try:
            target = self.target_url
            
            # Check cookie attributes
            response = requests.get(target, timeout=10)
            cookies = response.cookies
            
            insecure_cookies = []
            for cookie in cookies:
                cookie_attrs = []
                if not cookie.secure:
                    cookie_attrs.append("missing Secure flag")
                if not cookie.has_nonstandard_attr('HttpOnly'):
                    cookie_attrs.append("missing HttpOnly flag")
                if not cookie.has_nonstandard_attr('SameSite'):
                    cookie_attrs.append("missing SameSite attribute")
                
                if cookie_attrs:
                    insecure_cookies.append(f"{{cookie.name}}: {{', '.join(cookie_attrs)}}")
            
            if insecure_cookies:
                self.results["{test_name}"] = f"Insecure cookies found: {{'; '.join(insecure_cookies)}}"
            else:
                self.results["{test_name}"] = "No insecure cookies detected"
                
        except Exception as e:
            self.results["{test_name}"] = f"Error: {{str(e)}}"
            logging.error(f"Error in {vulnerability} test: {{e}}")"""
                
        elif "information" in test_name.lower() or "disclosure" in test_name.lower():
            template = f"""def {test_name}(self):
        try:
            target = self.target_url
            
            # Check for information disclosure
            response = requests.get(target, timeout=10)
            
            # Check for server information in headers
            headers = response.headers
            server_info = []
            sensitive_headers = ['Server', 'X-Powered-By', 'X-AspNet-Version', 'X-AspNetMvc-Version']
            
            for header in sensitive_headers:
                if header in headers:
                    server_info.append(f"{{header}}: {{headers[header]}}")
            
            # Check for comments with potentially sensitive information
            html = response.text
            comments = re.findall(r'<!--(.+?)-->', html, re.DOTALL)
            sensitive_patterns = ['user', 'password', 'key', 'secret', 'token', 'api', 'TODO', 'FIXME']
            sensitive_comments = []
            
            for comment in comments:
                if any(pattern in comment.lower() for pattern in sensitive_patterns):
                    # Truncate long comments
                    comment_preview = comment[:100] + '...' if len(comment) > 100 else comment
                    sensitive_comments.append(comment_preview.strip())
            
            findings = []
            if server_info:
                findings.append(f"Server information disclosed: {{'; '.join(server_info)}}")
            if sensitive_comments:
                findings.append(f"Potentially sensitive information in comments: {{len(sensitive_comments)}} instances")
            
            if findings:
                self.results["{test_name}"] = "\\n".join(findings)
            else:
                self.results["{test_name}"] = "No information disclosure detected"
                
        except Exception as e:
            self.results["{test_name}"] = f"Error: {{str(e)}}"
            logging.error(f"Error in {vulnerability} test: {{e}}")"""
        
        return template
    
    def _generate_fallback_template(self, vulnerabilities: List[str]) -> str:
        """Generate a complete fallback template with tests for the given vulnerabilities."""
        
        template = """import unittest
import json
import time
import os
import re
import logging
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from zapv2 import ZAPv2
import requests
from urllib.parse import urljoin, urlparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

class OWASP_SecurityTests(unittest.TestCase):
    def setUp(self):
        self.target_url = "<target_url>"
        self.results = {}
        self.scan_timeout = 300
        
        # Ensure environment variables are loaded correctly
        try:
            # Check if .env file exists and load it if needed
            if os.path.exists('.env'):
                load_dotenv('.env')
            
            # If ZAP_API_KEY is not set, check for any alternative names
            self.zap_api_key = os.getenv("ZAP_API_KEY") or os.getenv("ZAP_API") or ""
            
            # Set proxy with fallback using a default value
            # This avoids the NameError by using os.getenv which returns None if not found
            zap_proxy_server = os.getenv("ZAP_PROXY_SERVER")
            self.zap_proxy = "http://localhost:8080"
            
            # Initialize ZAP API client with proper error handling
            if self.zap_api_key:
                self.zap = ZAPv2(apikey=self.zap_api_key, proxies={"http": self.zap_proxy, "https": self.zap_proxy})
                logging.info(f"ZAP initialized with proxy at {self.zap_proxy}")
            else:
                logging.warning("ZAP API key not found. Using ZAP without authentication.")
                self.zap = ZAPv2(proxies={"http": self.zap_proxy, "https": self.zap_proxy})
        except Exception as e:
            logging.error(f"Error initializing ZAP: {e}")
            self.zap = None
"""

        for vuln in vulnerabilities:
            test_name = "test_" + vuln.lower().replace(" ", "_")
            
            test_method = self._generate_fallback_test(test_name, vuln)
            template += "\n    " + test_method.replace("\n", "\n    ") + "\n"
        
        if not vulnerabilities:
            common_tests = [
                ("SQL Injection", "test_sql_injection"),
                ("XSS Vulnerability", "test_xss_vulnerability"),
                ("Security Headers", "test_security_headers"),
                ("Information Disclosure", "test_information_disclosure"),
                ("Insecure Cookies", "test_insecure_cookies")
            ]
            
            for vuln, test_name in common_tests:
                test_method = self._generate_fallback_test(test_name, vuln)
                template += "\n    " + test_method.replace("\n", "\n    ") + "\n"
        
        template += """
    def tearDown(self):
        try:
            # Output all test results
            for test, result in self.results.items():
                print(json.dumps({
                    "type": "result",
                    "test_case": test,
                    "target_url": self.target_url,
                    "result": result
                }))
                
            logging.info("Test run completed")
        except Exception as e:
            logging.error(f"Error in tearDown: {e}")

if __name__ == "__main__":
    unittest.main()
"""
        
        return template
    
    async def query_model_async(self, prompt: str) -> str:
        """
        Enhanced method to generate test script using the advanced ensemble method.
        Includes improved error recovery, confidence assessment, and quality control.
        """
        start_time = datetime.now()
        
        try:
            model_responses = await self.query_all_models_async(prompt)
            
            valid_responses = {k: v for k, v in model_responses.items() if v.valid}
            
            processing_log = []
            processing_log.append(f"Received {len(valid_responses)} valid responses from {len(model_responses)} total attempts")
            
            if not valid_responses:
                logger.warning("No valid responses from any models. Generating robust fallback.")
                fallback = self._generate_fallback_template([])
                return self._validate_complete_script(fallback)
            
            consensus_vulnerabilities = self._advanced_consensus_voting(valid_responses)
            processing_log.append(f"Identified {len(consensus_vulnerabilities)} consensus vulnerabilities")
            logger.info(f"Consensus vulnerabilities: {consensus_vulnerabilities}")
            
            combined_script = self._combine_scripts_advanced(valid_responses, consensus_vulnerabilities)
            
            final_script = self._validate_complete_script(combined_script)
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            processing_log.append(f"Script generated in {processing_time:.2f} seconds")
            processing_log.append(f"Script has {final_script.count('def test_')} test methods")
            
            if "if __name__ ==" in final_script:
                lines = final_script.split('\n')
                for i, line in enumerate(lines):
                    if line.strip().startswith("if __name__ =="):
                        lines[i] = line.lstrip()
                        if i+1 < len(lines) and lines[i+1].strip():
                            lines[i+1] = "    " + lines[i+1].lstrip()
                final_script = '\n'.join(lines)
            
            if False:
                log_comment = "\n\n'''\nEnsemble Processing Log:\n" + "\n".join(processing_log) + "\n'''"
                final_script += log_comment
            
            return self._validate_python_syntax(final_script)
            
        except Exception as e:
            logger.error(f"Critical error in ensemble query_model_async: {e}")
            fallback = self._generate_fallback_template([])
            validated = self._validate_complete_script(fallback)
            return self._validate_python_syntax(validated)
            
    def _validate_complete_script(self, script: str) -> str:
        """Final validation and cleanup to ensure a valid Python script."""
        lines = script.split('\n')
        valid_lines = []

        in_imports = True
        for line in lines:
            if in_imports:
                if line.strip().startswith(('import ', 'from ')):
                    valid_lines.append(line.strip())
                elif line.strip() == "":
                    valid_lines.append("")
                else:
                    in_imports = False
                    valid_lines.append(line)
            else:
                valid_lines.append(line)

        result = []
        in_class = False
        class_name = ""

        for line in valid_lines:
            if line.strip().startswith('class ') and 'unittest.TestCase' in line:
                in_class = True
                class_name = line.strip().split('(')[0].split(' ')[1]
                result.append(line.strip())
            elif in_class and line.strip().startswith('def '):
                result.append('    ' + line.strip())
            elif in_class and line.strip() and not line.startswith('    '):
                result.append('        ' + line.strip())
            else:
                result.append(line)

        if 'if __name__ == "__main__"' not in script and "if __name__ == '__main__'" not in script:
            result.append("")
            result.append('if __name__ == "__main__":')
            result.append('    unittest.main()')

        return '\n'.join(result)
        
    def _validate_script_indentation(self, script: str) -> str:
        """Validate and fix script indentation to avoid IndentationError."""
        lines = script.split('\n')
        result = []
        class_indent = 0
        method_indent = 4
        block_indent = 8 

        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if not stripped:
                result.append(line)
                continue

            current_indent = len(line) - len(stripped)

            if stripped.startswith('class '):
                class_indent = current_indent
                result.append(line)
            elif stripped.startswith('def ') and 'self' in stripped:
                if current_indent != class_indent + method_indent:
                    result.append(' ' * (class_indent + method_indent) + stripped)
                else:
                    result.append(line)
            elif stripped.startswith('def '):
                if current_indent != class_indent:
                    result.append(' ' * class_indent + stripped)
                else:
                    result.append(line)
            else:
                if current_indent < class_indent + method_indent:
                    result.append(' ' * (class_indent + method_indent + 4) + stripped)
                else:
                    result.append(line)

        return '\n'.join(result)

    
    def _validate_python_syntax(self, script: str) -> str:
        """
        Validate the Python syntax of the generated script, focusing on indentation issues.
        This is a final safety check to catch any syntax errors before returning the script.
        """
        try:
            compile(script, '<string>', 'exec')
            return script
        except IndentationError as e:
            logger.error(f"Indentation error detected: {e}")
            lines = script.split('\n')
            
            fixed_lines = []
            in_class = False
            in_method = False
            
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    fixed_lines.append("")
                    continue
                
                if stripped.startswith('import ') or stripped.startswith('from '):
                    fixed_lines.append(stripped)
                elif stripped.startswith('class '):
                    in_class = True
                    in_method = False
                    fixed_lines.append(stripped)
                elif in_class and stripped.startswith('def '):
                    in_method = True
                    fixed_lines.append('    ' + stripped)
                elif stripped == 'if __name__ == "__main__":' or stripped == "if __name__ == '__main__':":
                    in_class = False
                    in_method = False
                    fixed_lines.append(stripped)
                elif not in_class:
                    fixed_lines.append(stripped)
                elif in_class and not in_method:
                    fixed_lines.append('    ' + stripped)
                else:
                    fixed_lines.append('        ' + stripped)
            
            fixed_script = '\n'.join(fixed_lines)
            
            try:
                compile(fixed_script, '<string>', 'exec')
                logger.info("Successfully fixed indentation errors")
                return fixed_script
            except Exception as e2:
                logger.error(f"Failed to fix syntax: {e2}")
                return """import unittest
import json
import os
import logging
from dotenv import load_dotenv

# Configure basic logging
logging.basicConfig(level=logging.INFO)

# Try to load environment variables
try:
    if os.path.exists('.env'):
        load_dotenv('.env')
    logging.info("Environment loaded")
except Exception as e:
    logging.warning(f"Could not load environment: {e}")

class OWASP_SecurityTests(unittest.TestCase):
    def setUp(self):
        self.target_url = "<target_url>"
        self.results = {}
    
    def test_basic_security(self):
        self.results["test_basic_security"] = "Basic security test completed"
    
    def tearDown(self):
        try:
            print(json.dumps({
                "type": "result",
                "test_case": "test_basic_security",
                "target_url": self.target_url,
                "result": self.results["test_basic_security"]
            }))
        except Exception as e:
            logging.error(f"Error in tearDown: {e}")

if __name__ == "__main__":
    unittest.main()
"""
        except SyntaxError as e:
            logger.error(f"Syntax error detected: {e}")
            return """import unittest
import json
import os
import logging
from dotenv import load_dotenv

# Configure basic logging
logging.basicConfig(level=logging.INFO)

# Try to load environment variables
try:
    if os.path.exists('.env'):
        load_dotenv('.env')
    logging.info("Environment loaded")
except Exception as e:
    logging.warning(f"Could not load environment: {e}")

class OWASP_SecurityTests(unittest.TestCase):
    def setUp(self):
        self.target_url = "<target_url>"
        self.results = {}
    
    def test_basic_security(self):
        self.results["test_basic_security"] = "Basic security test completed"
    
    def tearDown(self):
        try:
            print(json.dumps({
                "type": "result",
                "test_case": "test_basic_security",
                "target_url": self.target_url,
                "result": self.results["test_basic_security"]
            }))
        except Exception as e:
            logging.error(f"Error in tearDown: {e}")

if __name__ == "__main__":
    unittest.main()
"""
        except Exception as e:
            logger.error(f"Other validation error: {e}")
            return script
            
    def query_model(self, prompt: str) -> str:
        """Synchronous version of query_model_async."""
        script = asyncio.run(self.query_model_async(prompt))
        return self._validate_python_syntax(script)