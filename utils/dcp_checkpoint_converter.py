"""
DCP Checkpoint转换器
将PyTorch DCP格式的checkpoint转换为标准torch.save格式，用于推理
"""

import torch
import torch.distributed.checkpoint as dcp
from pathlib import Path
import logging
from typing import Dict, Any, Optional, Union
import shutil
import tempfile

logger = logging.getLogger(__name__)


class DCPCheckpointConverter:
    """
    DCP Checkpoint转换器
    
    将分布式训练保存的DCP格式checkpoint转换为推理友好的torch.save格式
    """
    
    def __init__(self, verbose: bool = True):
        """
        初始化转换器
        
        Args:
            verbose: 是否显示详细日志
        """
        self.verbose = verbose
        if self.verbose:
            logging.basicConfig(level=logging.INFO)
    
    def convert(
        self, 
        dcp_checkpoint_path: Union[str, Path],
        output_path: Union[str, Path],
        include_optimizer: bool = False,
        include_scheduler: bool = False,
        model_only: bool = False
    ) -> bool:
        """
        转换DCP checkpoint到torch.save格式
        
        Args:
            dcp_checkpoint_path: DCP checkpoint目录路径
            output_path: 输出的.pth文件路径
            include_optimizer: 是否包含优化器状态
            include_scheduler: 是否包含调度器状态
            model_only: 是否只保存模型权重（推理推荐）
            
        Returns:
            bool: 转换是否成功
        """
        try:
            dcp_path = Path(dcp_checkpoint_path)
            output_path = Path(output_path)
            
            if not dcp_path.exists():
                logger.error(f"DCP checkpoint路径不存在: {dcp_path}")
                return False
            
            # 确保输出目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"开始转换DCP checkpoint...")
            logger.info(f"源路径: {dcp_path}")
            logger.info(f"目标路径: {output_path}")
            
            # 加载DCP checkpoint
            state_dict = self._load_dcp_checkpoint(dcp_path)
            if state_dict is None:
                return False
            
            # 构建输出状态字典
            output_state = self._build_output_state(
                state_dict, 
                include_optimizer=include_optimizer,
                include_scheduler=include_scheduler,
                model_only=model_only
            )
            
            # 保存为torch.save格式
            self._save_torch_checkpoint(output_state, output_path)
            
            # 验证转换结果
            if self._verify_converted_checkpoint(output_path):
                logger.info(f"✅ DCP checkpoint转换成功!")
                self._print_checkpoint_info(output_state, output_path)
                return True
            else:
                logger.error("❌ 转换后的checkpoint验证失败")
                return False
                
        except Exception as e:
            logger.error(f"❌ DCP checkpoint转换失败: {e}")
            import traceback
            logger.error(f"详细错误: {traceback.format_exc()}")
            return False
    
    def _load_dcp_checkpoint(self, dcp_path: Path) -> Optional[Dict[str, Any]]:
        """加载DCP checkpoint"""
        try:
            logger.info("📥 加载DCP checkpoint...")
            
            # 检查DCP文件
            distcp_files = list(dcp_path.glob("*.distcp"))
            if not distcp_files:
                logger.error(f"在 {dcp_path} 中未找到.distcp文件")
                return None
            
            logger.info(f"找到 {len(distcp_files)} 个DCP文件")
            
            # 使用PyTorch DCP的format_utils进行转换
            # 这是官方推荐的DCP→torch.save转换方法
            logger.info("使用format_utils转换DCP checkpoint...")
            
            import tempfile
            from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            # 转换DCP到torch.save格式
            dcp_to_torch_save(str(dcp_path), temp_path)
            
            # 加载转换后的文件 - 处理OmegaConf问题
            try:
                # 先尝试安全加载
                state_dict = torch.load(temp_path, map_location='cpu', weights_only=True)
            except Exception:
                logger.info("检测到OmegaConf类型，使用兼容模式加载...")
                # 添加OmegaConf到安全全局列表
                torch.serialization.add_safe_globals([
                    'omegaconf.listconfig.ListConfig', 
                    'omegaconf.dictconfig.DictConfig'
                ])
                state_dict = torch.load(temp_path, map_location='cpu', weights_only=False)
            
            # 清理临时文件
            import os
            os.unlink(temp_path)
            
            logger.info(f"✅ DCP checkpoint加载成功")
            logger.info(f"包含的键: {list(state_dict.keys())}")
            
            # 显示详细信息
            for key, value in state_dict.items():
                if isinstance(value, dict):
                    logger.info(f"  🔑 {key}: {len(value)} 个子项")
                else:
                    logger.info(f"  🔑 {key}: {type(value).__name__}")
            
            return state_dict
            
        except Exception as e:
            logger.error(f"加载DCP checkpoint失败: {e}")
            import traceback
            logger.error(f"详细错误: {traceback.format_exc()}")
            return None
    
    def _build_output_state(
        self, 
        dcp_state: Dict[str, Any],
        include_optimizer: bool = False,
        include_scheduler: bool = False,
        model_only: bool = False
    ) -> Dict[str, Any]:
        """构建输出状态字典"""
        logger.info("🔧 构建输出状态字典...")
        
        output_state = {}
        
        # 1. 模型权重（必需）
        if "model" in dcp_state:
            output_state["model"] = dcp_state["model"]
            logger.info(f"✅ 包含模型权重: {len(dcp_state['model'])} 个参数")
        else:
            logger.warning("⚠️ 未找到模型权重")
        
        # 2. EMA权重（推理重要）
        if "ema" in dcp_state:
            output_state["ema"] = dcp_state["ema"]
            logger.info("✅ 包含EMA权重")
        else:
            logger.info("ℹ️ 未找到EMA权重")
        
        # 如果只要模型，跳过其他组件
        if model_only:
            logger.info("📦 模型专用模式：只保留模型和EMA权重")
            return output_state
        
        # 3. 优化器状态（可选）
        if include_optimizer and "optimizer" in dcp_state:
            output_state["optimizer"] = dcp_state["optimizer"]
            logger.info("✅ 包含优化器状态")
        
        # 4. 调度器状态（可选）
        if include_scheduler and "scheduler" in dcp_state:
            output_state["scheduler"] = dcp_state["scheduler"]
            logger.info("✅ 包含调度器状态")
        
        # 5. 训练元数据
        metadata = {}
        for key, value in dcp_state.items():
            if key not in ["model", "optimizer", "ema", "scheduler"]:
                if isinstance(value, (int, float, str, bool)):
                    metadata[key] = value
        
        if metadata:
            output_state["metadata"] = metadata
            logger.info(f"✅ 包含元数据: {list(metadata.keys())}")
        
        return output_state
    
    def _save_torch_checkpoint(self, state_dict: Dict[str, Any], output_path: Path):
        """保存为torch.save格式"""
        logger.info(f"💾 保存为torch.save格式...")
        
        # 使用临时文件确保原子性写入
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
            torch.save(state_dict, tmp_file.name)
            tmp_path = tmp_file.name
        
        # 移动到最终位置
        shutil.move(tmp_path, output_path)
        logger.info(f"✅ 保存完成: {output_path}")
    
    def _verify_converted_checkpoint(self, checkpoint_path: Path) -> bool:
        """验证转换后的checkpoint"""
        try:
            logger.info("🔍 验证转换后的checkpoint...")
            
            # 尝试加载 - 处理OmegaConf问题
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            except Exception:
                # 如果有OmegaConf，使用兼容模式
                torch.serialization.add_safe_globals([
                    'omegaconf.listconfig.ListConfig', 
                    'omegaconf.dictconfig.DictConfig'
                ])
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # 检查基本结构
            if not isinstance(checkpoint, dict):
                logger.error("Checkpoint不是字典格式")
                return False
            
            # 检查模型权重
            if "model" not in checkpoint:
                logger.error("缺少模型权重")
                return False
            
            model_params = checkpoint["model"]
            if not isinstance(model_params, dict) or len(model_params) == 0:
                logger.error("模型权重为空或格式错误")
                return False
            
            logger.info(f"✅ Checkpoint验证通过")
            return True
            
        except Exception as e:
            logger.error(f"验证checkpoint失败: {e}")
            import traceback
            logger.error(f"验证详细错误: {traceback.format_exc()}")
            return False
    
    def _print_checkpoint_info(self, state_dict: Dict[str, Any], output_path: Path):
        """打印checkpoint信息"""
        logger.info("\n" + "="*50)
        logger.info("📊 转换后的Checkpoint信息:")
        logger.info("="*50)
        
        # 文件大小
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"📁 文件大小: {file_size_mb:.1f} MB")
        
        # 组件信息
        components = []
        if "model" in state_dict:
            model_params = len(state_dict["model"])
            components.append(f"模型参数: {model_params:,} 个")
        
        if "ema" in state_dict:
            components.append("EMA权重: ✅")
        
        if "optimizer" in state_dict:
            components.append("优化器状态: ✅")
        
        if "metadata" in state_dict:
            metadata = state_dict["metadata"]
            if "step" in metadata:
                components.append(f"训练步数: {metadata['step']}")
        
        for component in components:
            logger.info(f"📦 {component}")
        
        logger.info("="*50)
        logger.info(f"✅ 可直接用于推理: torch.load('{output_path}')")
        logger.info("="*50 + "\n")


def convert_dcp_checkpoint(
    dcp_path: str,
    output_path: str,
    model_only: bool = True,
    include_optimizer: bool = False,
    include_scheduler: bool = False,
    verbose: bool = True
) -> bool:
    """
    便捷函数：转换DCP checkpoint
    
    Args:
        dcp_path: DCP checkpoint目录路径
        output_path: 输出.pth文件路径
        model_only: 是否只保存模型权重（推荐推理使用）
        include_optimizer: 是否包含优化器状态
        include_scheduler: 是否包含调度器状态
        verbose: 是否显示详细日志
    
    Returns:
        bool: 是否转换成功
    """
    converter = DCPCheckpointConverter(verbose=verbose)
    return converter.convert(
        dcp_checkpoint_path=dcp_path,
        output_path=output_path,
        include_optimizer=include_optimizer,
        include_scheduler=include_scheduler,
        model_only=model_only
    )


if __name__ == "__main__":
    # 测试转换
    import sys
    
    if len(sys.argv) < 3:
        print("用法: python dcp_checkpoint_converter.py <dcp_path> <output_path>")
        sys.exit(1)
    
    dcp_path = sys.argv[1]
    output_path = sys.argv[2]
    
    success = convert_dcp_checkpoint(dcp_path, output_path)
    sys.exit(0 if success else 1)
