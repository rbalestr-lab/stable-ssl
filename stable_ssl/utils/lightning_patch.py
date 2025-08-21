"""Monkey-patch for PyTorch Lightning to support manual optimization with Trainer parameters.

This patch modifies Lightning's validation to transfer gradient_clip_val and 
accumulate_grad_batches to alternative attributes instead of raising errors.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import lightning.pytorch as pl


def apply_manual_optimization_patch():
    """Apply the monkey-patch to Lightning's manual optimization validation.
    
    This patch modifies the __verify_manual_optimization_support function to:
    1. Transfer gradient_clip_val to gradient_clip_val_ 
    2. Transfer accumulate_grad_batches to accumulate_grad_batches_
    3. Clear the original values to avoid Lightning's error
    
    This allows users to use standard Trainer parameters even with manual optimization.
    """
    try:
        # Try new import path first (lightning.pytorch)
        try:
            import lightning.pytorch.trainer.configuration_validator as validator_module
        except ImportError:
            # Fall back to old import path (pytorch_lightning)
            import pytorch_lightning.trainer.configuration_validator as validator_module
        
        # Store the original function for potential restoration
        original_verify = validator_module.__verify_manual_optimization_support
        
        def patched_verify_manual_optimization_support(trainer: "pl.Trainer", model: "pl.LightningModule") -> None:
            """Patched version that transfers parameters instead of raising errors."""
            # Only process if manual optimization is enabled
            if not model.automatic_optimization:
                # Transfer gradient clipping parameters
                if trainer.gradient_clip_val is not None and trainer.gradient_clip_val > 0:
                    # Save to alternative attributes
                    trainer.gradient_clip_val_ = trainer.gradient_clip_val
                    trainer.gradient_clip_algorithm_ = trainer.gradient_clip_algorithm
                    
                    # Clear originals to avoid Lightning's validation error
                    trainer.gradient_clip_val = None
                    
                    logging.debug(
                        f"Manual optimization patch: Transferred gradient_clip_val={trainer.gradient_clip_val_} "
                        f"to trainer.gradient_clip_val_ (algorithm={trainer.gradient_clip_algorithm_})"
                    )
                
                # Transfer gradient accumulation parameters
                if trainer.accumulate_grad_batches != 1:
                    # Save to alternative attribute
                    trainer.accumulate_grad_batches_ = trainer.accumulate_grad_batches
                    
                    # Reset to 1 to avoid Lightning's validation error
                    trainer.accumulate_grad_batches = 1
                    
                    logging.debug(
                        f"Manual optimization patch: Transferred accumulate_grad_batches={trainer.accumulate_grad_batches_} "
                        f"to trainer.accumulate_grad_batches_"
                    )
            
            # No need to call the original since we handled the problematic cases
            # The original would only raise errors that we're avoiding
            return None
        
        # Apply the monkey-patch
        validator_module.__verify_manual_optimization_support = patched_verify_manual_optimization_support
        
        # Store reference to original for potential restoration
        validator_module.__original_verify_manual_optimization_support = original_verify
        
        logging.debug("Successfully applied manual optimization parameter patch for PyTorch Lightning")
        
    except ImportError as e:
        logging.warning(f"Could not apply Lightning patch: {e}")
    except Exception as e:
        logging.warning(f"Error applying Lightning patch: {e}")


def restore_original_validation():
    """Restore the original Lightning validation function (for testing/debugging)."""
    try:
        # Try new import path first
        try:
            import lightning.pytorch.trainer.configuration_validator as validator_module
        except ImportError:
            import pytorch_lightning.trainer.configuration_validator as validator_module
        
        if hasattr(validator_module, '__original_verify_manual_optimization_support'):
            validator_module.__verify_manual_optimization_support = \
                validator_module.__original_verify_manual_optimization_support
            delattr(validator_module, '__original_verify_manual_optimization_support')
            logging.debug("Restored original Lightning validation function")
        else:
            logging.warning("No original validation function found to restore")
            
    except ImportError:
        pass