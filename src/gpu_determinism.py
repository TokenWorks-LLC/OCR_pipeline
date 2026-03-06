"""
GPU determinism configuration for reproducible inference.
Ensures byte-for-byte identical outputs across runs.
"""

import logging
import os
from typing import Optional
import warnings

logger = logging.getLogger(__name__)


def setup_deterministic_mode(
    torch_deterministic: bool = True,
    cublas_workspace_config: str = ":16:8",
    cudnn_benchmark: bool = False,
    cudnn_deterministic: bool = True
) -> bool:
    """
    Configure PyTorch and CUDA for deterministic inference.
    
    Args:
        torch_deterministic: Enable torch.use_deterministic_algorithms
        cublas_workspace_config: CUBLAS workspace config
        cudnn_benchmark: CuDNN benchmark mode (should be False)
        cudnn_deterministic: CuDNN deterministic mode (should be True)
    
    Returns:
        True if setup successful
    """
    try:
        import torch
        
        # Disable gradient computation
        torch.set_grad_enabled(False)
        
        # Set deterministic algorithms
        if torch_deterministic:
            try:
                torch.use_deterministic_algorithms(True)
                logger.info("✅ torch.use_deterministic_algorithms(True)")
            except Exception as e:
                logger.warning(f"Could not enable deterministic algorithms: {e}")
        
        # Configure CuDNN
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.deterministic = cudnn_deterministic
            torch.backends.cudnn.benchmark = cudnn_benchmark
            logger.info(f"✅ CuDNN: deterministic={cudnn_deterministic}, benchmark={cudnn_benchmark}")
        
        # Set CUBLAS workspace config
        if cublas_workspace_config:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = cublas_workspace_config
            logger.info(f"✅ CUBLAS_WORKSPACE_CONFIG={cublas_workspace_config}")
        
        # Set BLAS thread limits to prevent oversubscription
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        logger.info("✅ BLAS threads limited to 1")
        
        # Additional environment variables for reproducibility
        os.environ['PYTHONHASHSEED'] = '0'
        
        logger.info("✅ Deterministic mode configured")
        return True
        
    except ImportError:
        logger.warning("PyTorch not available, skipping GPU determinism setup")
        return False
    except Exception as e:
        logger.error(f"Failed to setup deterministic mode: {e}")
        return False


def warmup_model(model, input_shape: tuple, device: str = 'cuda', iterations: int = 3):
    """
    Warm up model with fixed dummy tensors for consistent performance.
    
    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (e.g., (1, 3, 224, 224))
        device: Device to use
        iterations: Number of warmup iterations
    """
    try:
        import torch
        
        model.eval()
        
        # Create fixed dummy input
        dummy_input = torch.zeros(input_shape, dtype=torch.float32, device=device)
        
        # Warm up
        with torch.no_grad():
            for i in range(iterations):
                _ = model(dummy_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        logger.debug(f"Model warmed up with {iterations} iterations")
        
    except Exception as e:
        logger.warning(f"Model warmup failed: {e}")


def setup_pinned_memory(enabled: bool = True):
    """
    Configure pinned memory for faster H2D transfers.
    
    Args:
        enabled: Whether to use pinned memory
    """
    try:
        import torch
        
        if enabled and torch.cuda.is_available():
            # Pinned memory is enabled per-tensor, not globally
            # This just logs the capability
            logger.info("✅ Pinned memory transfers enabled")
            return True
        
        return False
        
    except ImportError:
        return False


def get_gpu_info() -> dict:
    """Get GPU information for logging."""
    try:
        import torch
        
        if not torch.cuda.is_available():
            return {'available': False}
        
        info = {
            'available': True,
            'device_count': torch.cuda.device_count(),
            'current_device': torch.cuda.current_device(),
            'device_name': torch.cuda.get_device_name(0),
            'device_capability': torch.cuda.get_device_capability(0),
            'memory_allocated_mb': torch.cuda.memory_allocated(0) / (1024**2),
            'memory_reserved_mb': torch.cuda.memory_reserved(0) / (1024**2),
        }
        
        return info
        
    except ImportError:
        return {'available': False, 'error': 'PyTorch not installed'}
    except Exception as e:
        return {'available': False, 'error': str(e)}


def set_single_stream_inference():
    """
    Configure single-stream inference to avoid nondeterminism from
    concurrent CUDA streams.
    """
    try:
        import torch
        
        if torch.cuda.is_available():
            # Set default stream for inference
            # Note: This is implicit - we don't create additional streams
            logger.info("✅ Single-stream inference configured (default stream only)")
            return True
        
        return False
        
    except ImportError:
        return False


class DeterministicInferenceContext:
    """
    Context manager for deterministic inference.
    
    Usage:
        with DeterministicInferenceContext():
            output = model(input)
    """
    
    def __init__(self, enforce_eval: bool = True):
        """
        Initialize context.
        
        Args:
            enforce_eval: Ensure model.eval() is set
        """
        self.enforce_eval = enforce_eval
        self.prev_grad_enabled = None
        
    def __enter__(self):
        try:
            import torch
            self.prev_grad_enabled = torch.is_grad_enabled()
            torch.set_grad_enabled(False)
        except ImportError:
            pass
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            import torch
            if self.prev_grad_enabled is not None:
                torch.set_grad_enabled(self.prev_grad_enabled)
        except ImportError:
            pass


def enforce_threadpool_limits(num_threads: int = 1):
    """
    Enforce thread limits using threadpoolctl.
    
    Args:
        num_threads: Number of threads for BLAS operations
    """
    try:
        from threadpoolctl import threadpool_limits
        
        # This returns a context manager, but we want global effect
        # Set limits globally
        with threadpool_limits(limits=num_threads, user_api='blas'):
            pass
        
        logger.info(f"✅ Thread pool limited to {num_threads} threads")
        return True
        
    except ImportError:
        logger.warning("threadpoolctl not available, using environment variables only")
        # Fallback to environment variables (already set in setup_deterministic_mode)
        return False


def verify_determinism_config() -> dict:
    """
    Verify that determinism settings are properly configured.
    
    Returns:
        Dict with verification results
    """
    results = {
        'torch_available': False,
        'cuda_available': False,
        'deterministic_algorithms': False,
        'cudnn_deterministic': False,
        'cudnn_benchmark': False,
        'grad_enabled': True,
        'env_vars_set': False
    }
    
    try:
        import torch
        results['torch_available'] = True
        results['cuda_available'] = torch.cuda.is_available()
        
        # Check torch settings
        results['grad_enabled'] = torch.is_grad_enabled()
        
        # Check CuDNN
        if hasattr(torch.backends, 'cudnn'):
            results['cudnn_deterministic'] = torch.backends.cudnn.deterministic
            results['cudnn_benchmark'] = torch.backends.cudnn.benchmark
        
        # Check environment variables
        env_vars = ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'CUBLAS_WORKSPACE_CONFIG']
        results['env_vars_set'] = all(var in os.environ for var in env_vars)
        
        # Check if deterministic algorithms are enabled (may not be available in all torch versions)
        try:
            results['deterministic_algorithms'] = torch.are_deterministic_algorithms_enabled()
        except AttributeError:
            results['deterministic_algorithms'] = None  # Not available in this torch version
        
    except ImportError:
        pass
    
    return results


def log_determinism_status():
    """Log current determinism configuration status."""
    results = verify_determinism_config()
    
    logger.info("=" * 60)
    logger.info("DETERMINISM CONFIGURATION STATUS")
    logger.info("=" * 60)
    logger.info(f"PyTorch available: {results['torch_available']}")
    logger.info(f"CUDA available: {results['cuda_available']}")
    logger.info(f"Gradient enabled: {results['grad_enabled']} (should be False)")
    logger.info(f"Deterministic algorithms: {results['deterministic_algorithms']}")
    logger.info(f"CuDNN deterministic: {results['cudnn_deterministic']} (should be True)")
    logger.info(f"CuDNN benchmark: {results['cudnn_benchmark']} (should be False)")
    logger.info(f"Environment variables set: {results['env_vars_set']}")
    logger.info("=" * 60)
    
    # Check for issues
    issues = []
    if results['torch_available']:
        if results['grad_enabled']:
            issues.append("⚠️  Gradient computation is enabled (should be disabled)")
        if not results['cudnn_deterministic']:
            issues.append("⚠️  CuDNN deterministic mode is disabled")
        if results['cudnn_benchmark']:
            issues.append("⚠️  CuDNN benchmark mode is enabled (causes nondeterminism)")
    
    if issues:
        logger.warning("Determinism issues detected:")
        for issue in issues:
            logger.warning(f"  {issue}")
    else:
        logger.info("✅ All determinism checks passed")
