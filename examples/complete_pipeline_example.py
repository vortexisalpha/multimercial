"""
Complete Video Advertisement Placement Pipeline Example

This example demonstrates the full capabilities of the video processing pipeline
including configuration, processing, monitoring, cloud integration, and deployment.
"""

import asyncio
import torch
import numpy as np
import logging
from pathlib import Path
import sys
import time

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from video_ad_placement.pipeline import (
    VideoAdPlacementPipeline, PipelineConfig, QualityLevel, ProcessingMode,
    PlacementConfig, CloudConfig, CloudProvider, ScalingPolicy
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Demonstrate the complete video processing pipeline."""
    
    logger.info("=== Video Advertisement Placement Pipeline Demo ===")
    
    # 1. Configure the pipeline
    logger.info("1. Configuring pipeline...")
    
    pipeline_config = PipelineConfig(
        # Quality and performance settings
        quality_level=QualityLevel.HIGH,
        processing_mode=ProcessingMode.BATCH,
        max_workers=4,
        use_gpu=torch.cuda.is_available(),
        gpu_devices=[0] if torch.cuda.is_available() else [],
        
        # Memory management
        max_memory_usage=4096.0,  # 4GB
        batch_size=16,
        prefetch_frames=8,
        cache_size=500,
        
        # Error handling and recovery
        max_retries=3,
        retry_delay=1.0,
        checkpoint_interval=50,  # Checkpoint every 50 frames
        auto_recovery=True,
        fallback_quality=QualityLevel.MEDIUM,
        
        # Monitoring and logging
        enable_profiling=True,
        log_level="INFO",
        metrics_interval=2.0,
        health_check_interval=5.0,
        
        # Output configuration
        output_formats=["mp4"],
        output_qualities=["1080p"],
        intermediate_outputs=True,
        save_debug_data=True,
        
        # Cloud and distributed processing (disabled for local demo)
        enable_cloud_storage=False,
        distributed_processing=False
    )
    
    # TV placement configuration
    placement_config = PlacementConfig(
        target_wall_types=["wall", "background"],
        min_wall_area=1.5,  # 1.5 square meters minimum
        max_wall_distance=8.0,  # 8 meters maximum
        tv_width=1.0,  # 1 meter TV width
        tv_height=0.6,  # 0.6 meter TV height
        placement_stability_threshold=0.75,
        quality_threshold=0.7,
        temporal_smoothing=True,
        lighting_adaptation=True,
        occlusion_handling=True
    )
    
    logger.info(f"Pipeline configured with quality level: {pipeline_config.quality_level.value}")
    logger.info(f"GPU acceleration: {'enabled' if pipeline_config.use_gpu else 'disabled'}")
    
    # 2. Initialize the pipeline
    logger.info("2. Initializing pipeline components...")
    
    try:
        pipeline = VideoAdPlacementPipeline(pipeline_config)
        logger.info("✓ Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"✗ Failed to initialize pipeline: {e}")
        return
    
    # 3. Create sample advertisement content
    logger.info("3. Creating sample advertisement...")
    
    # Create a simple colored rectangle as advertisement (RGB format)
    ad_height, ad_width = 480, 640  # Standard resolution
    advertisement = torch.zeros((3, ad_height, ad_width), dtype=torch.float32)
    
    # Create a gradient pattern
    for i in range(ad_height):
        for j in range(ad_width):
            # Create a colorful gradient pattern
            r = (i / ad_height) * 0.8 + 0.2  # Red gradient
            g = (j / ad_width) * 0.8 + 0.2   # Green gradient
            b = 0.6  # Constant blue
            
            advertisement[0, i, j] = r  # Red channel
            advertisement[1, i, j] = g  # Green channel
            advertisement[2, i, j] = b  # Blue channel
    
    logger.info(f"✓ Advertisement created: {ad_width}x{ad_height} pixels")
    
    # 4. Prepare input and output paths
    logger.info("4. Setting up file paths...")
    
    # For demonstration, we'll create paths even if the input video doesn't exist
    input_video_path = "input_video.mp4"
    output_video_path = "output_with_ads.mp4"
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_video_path = str(output_dir / "processed_video.mp4")
    
    logger.info(f"Input video: {input_video_path}")
    logger.info(f"Output video: {output_video_path}")
    
    # 5. Estimate processing requirements
    logger.info("5. Estimating processing requirements...")
    
    # Create mock video info for estimation
    from video_ad_placement.pipeline.main_pipeline import VideoInfo
    mock_video_info = VideoInfo(
        path=input_video_path,
        duration=30.0,  # 30 seconds
        fps=30.0,
        width=1920,
        height=1080,
        total_frames=900,  # 30 seconds * 30 fps
        codec="h264",
        bitrate=5000000,  # 5 Mbps
        file_size=18750000,  # ~18.75 MB
        has_audio=True
    )
    
    estimate = pipeline.estimate_processing_time(mock_video_info)
    
    logger.info(f"✓ Processing estimate:")
    logger.info(f"  - Estimated time: {estimate.estimated_time:.1f} seconds")
    logger.info(f"  - Estimated memory: {estimate.estimated_memory:.1f} MB")
    logger.info(f"  - Estimated GPU memory: {estimate.estimated_gpu_memory:.1f} MB")
    logger.info(f"  - Bottleneck: {estimate.bottleneck_component}")
    logger.info(f"  - Confidence: {estimate.confidence:.1%}")
    
    if estimate.optimization_suggestions:
        logger.info("  - Optimization suggestions:")
        for suggestion in estimate.optimization_suggestions:
            logger.info(f"    * {suggestion}")
    
    # 6. Monitor pipeline health
    logger.info("6. Checking pipeline health...")
    
    stats = pipeline.get_processing_stats()
    logger.info(f"✓ Pipeline statistics:")
    logger.info(f"  - Videos processed: {stats['total_videos_processed']}")
    logger.info(f"  - Frames processed: {stats['total_frames_processed']}")
    logger.info(f"  - Total processing time: {stats['total_processing_time']:.1f}s")
    logger.info(f"  - Total errors: {stats['total_errors']}")
    
    # 7. Demonstrate streaming capabilities (simulation)
    logger.info("7. Demonstrating streaming capabilities...")
    
    try:
        from video_ad_placement.pipeline.streaming import VideoStream, StreamConfig, FrameData
        
        # Configure streaming
        stream_config = StreamConfig(
            buffer_size=20,
            max_buffer_memory_mb=256.0,
            target_fps=30.0,
            enable_threading=True,
            max_latency_ms=100.0
        )
        
        logger.info("✓ Streaming configuration created")
        logger.info(f"  - Buffer size: {stream_config.buffer_size} frames")
        logger.info(f"  - Max latency: {stream_config.max_latency_ms} ms")
        logger.info(f"  - Target FPS: {stream_config.target_fps}")
        
    except Exception as e:
        logger.warning(f"Streaming demo skipped: {e}")
    
    # 8. Demonstrate checkpointing
    logger.info("8. Demonstrating checkpointing...")
    
    try:
        checkpoint_stats = pipeline.checkpoint_manager.get_checkpoint_stats()
        logger.info("✓ Checkpoint system status:")
        logger.info(f"  - Strategy: {checkpoint_stats['strategy']}")
        logger.info(f"  - Interval: {checkpoint_stats['checkpoint_interval']} frames")
        logger.info(f"  - Total checkpoints: {checkpoint_stats['total_checkpoints']}")
        logger.info(f"  - Storage path: {checkpoint_stats['storage_path']}")
        
    except Exception as e:
        logger.warning(f"Checkpointing demo limited: {e}")
    
    # 9. Demonstrate resource management
    logger.info("9. Checking resource management...")
    
    try:
        resource_usage = pipeline.resource_manager.get_current_usage()
        logger.info("✓ Current resource usage:")
        logger.info(f"  - CPU usage: {resource_usage.cpu_usage_percent:.1f}%")
        logger.info(f"  - Memory usage: {resource_usage.memory_usage_mb:.1f} MB")
        logger.info(f"  - Memory available: {resource_usage.memory_available_mb:.1f} MB")
        
        if resource_usage.gpu_usage_percent:
            for gpu_id, usage in resource_usage.gpu_usage_percent.items():
                logger.info(f"  - GPU {gpu_id} usage: {usage:.1f}%")
                gpu_mem_used = resource_usage.gpu_memory_usage_mb.get(gpu_id, 0)
                gpu_mem_avail = resource_usage.gpu_memory_available_mb.get(gpu_id, 0)
                logger.info(f"  - GPU {gpu_id} memory: {gpu_mem_used:.1f} MB used, {gpu_mem_avail:.1f} MB available")
        
    except Exception as e:
        logger.warning(f"Resource monitoring limited: {e}")
    
    # 10. Demonstrate monitoring system
    logger.info("10. Checking monitoring system...")
    
    try:
        health_status = pipeline.monitor.get_health_status()
        logger.info("✓ System health status:")
        logger.info(f"  - Overall status: {health_status['overall_status']}")
        
        for component, status in health_status['components'].items():
            logger.info(f"  - {component}: {status['status']}")
            if status.get('error_count', 0) > 0:
                logger.info(f"    Errors: {status['error_count']}")
        
        alert_summary = health_status.get('alert_summary', {})
        if any(count > 0 for count in alert_summary.values()):
            logger.info("  - Recent alerts:")
            for level, count in alert_summary.items():
                if count > 0:
                    logger.info(f"    {level}: {count}")
        
    except Exception as e:
        logger.warning(f"Monitoring demo limited: {e}")
    
    # 11. Demonstrate cloud integration (local mode)
    logger.info("11. Demonstrating cloud integration...")
    
    try:
        from video_ad_placement.pipeline.cloud_integration import CloudConfig, CloudStorageManager, CloudProvider
        
        cloud_config = CloudConfig(
            provider=CloudProvider.LOCAL,  # Use local storage for demo
            storage_bucket="demo-bucket",
            enable_encryption=True,
            enable_distributed=False
        )
        
        storage_manager = CloudStorageManager(cloud_config)
        
        # Create a small test file
        test_file_path = output_dir / "test_file.txt"
        test_file_path.write_text("This is a test file for cloud storage demo.")
        
        # Upload test file
        upload_success = await storage_manager.upload_file(
            str(test_file_path),
            "demo/test_file.txt",
            metadata={"demo": "true", "timestamp": str(time.time())}
        )
        
        if upload_success:
            logger.info("✓ Cloud storage demo:")
            logger.info("  - File uploaded successfully")
            
            # List files
            files = await storage_manager.list_files("demo/")
            logger.info(f"  - Files in storage: {len(files)}")
            for file_info in files:
                logger.info(f"    * {file_info['key']} ({file_info['size']} bytes)")
            
            # Get transfer stats
            transfer_stats = storage_manager.get_transfer_stats()
            logger.info(f"  - Transfer statistics:")
            logger.info(f"    Uploads: {transfer_stats['uploads']}")
            logger.info(f"    Downloads: {transfer_stats['downloads']}")
            logger.info(f"    Bytes uploaded: {transfer_stats['bytes_uploaded']}")
            logger.info(f"    Errors: {transfer_stats['errors']}")
        
    except Exception as e:
        logger.warning(f"Cloud integration demo limited: {e}")
    
    # 12. Performance optimization suggestions
    logger.info("12. Performance optimization suggestions...")
    
    try:
        optimizations = pipeline.resource_manager.optimize_allocations()
        if optimizations:
            logger.info("✓ Optimization suggestions:")
            for suggestion in optimizations:
                logger.info(f"  - {suggestion}")
        else:
            logger.info("✓ No optimization suggestions - system running efficiently")
        
    except Exception as e:
        logger.warning(f"Optimization analysis limited: {e}")
    
    # 13. Simulated processing (since we may not have actual video file)
    logger.info("13. Simulating video processing...")
    
    try:
        # Check if input video exists
        if not Path(input_video_path).exists():
            logger.info("Input video not found - simulating processing workflow")
            
            # Simulate the processing steps
            logger.info("  - Video analysis: SIMULATED")
            logger.info("  - Depth estimation: SIMULATED") 
            logger.info("  - Object detection: SIMULATED")
            logger.info("  - Plane detection: SIMULATED")
            logger.info("  - TV placement: SIMULATED")
            logger.info("  - Temporal consistency: SIMULATED")
            logger.info("  - Video rendering: SIMULATED")
            
            # Create a dummy result
            from video_ad_placement.pipeline.main_pipeline import ProcessingResult, ResourceUsage
            
            simulated_result = ProcessingResult(
                output_path=output_video_path,
                processing_time=15.0,
                total_frames=900,
                successful_frames=895,
                failed_frames=5,
                quality_metrics={
                    'overall_quality': 0.85,
                    'temporal_consistency': 0.92,
                    'avg_processing_time': 0.017
                },
                resource_usage=ResourceUsage(
                    cpu_usage=65.0,
                    memory_usage=2048.0,
                    gpu_usage=45.0,
                    gpu_memory_usage=1024.0,
                    disk_io=50.0,
                    network_io=0.0,
                    processing_time=15.0,
                    peak_memory=2560.0,
                    energy_consumption=120.0
                ),
                error_log=["Frame 156: Low quality score", "Frame 789: Temporal discontinuity"]
            )
            
            logger.info("✓ Simulated processing completed:")
            logger.info(f"  - Success rate: {simulated_result.success_rate:.1%}")
            logger.info(f"  - Processing time: {simulated_result.processing_time:.1f} seconds")
            logger.info(f"  - Quality score: {simulated_result.quality_metrics['overall_quality']:.2f}")
            logger.info(f"  - Peak memory usage: {simulated_result.resource_usage.peak_memory:.1f} MB")
            
        else:
            # Process actual video
            logger.info("Processing actual video file...")
            result = await pipeline.process_video(
                input_video_path,
                output_video_path,
                advertisement,
                placement_config
            )
            
            logger.info("✓ Video processing completed:")
            logger.info(f"  - Success rate: {result.success_rate:.1%}")
            logger.info(f"  - Processing time: {result.processing_time:.1f} seconds")
            logger.info(f"  - Output saved to: {result.output_path}")
            
    except Exception as e:
        logger.error(f"Processing failed: {e}")
    
    # 14. Final system status
    logger.info("14. Final system status...")
    
    try:
        final_stats = pipeline.get_processing_stats()
        health_status = pipeline.monitor.get_health_status()
        
        logger.info("✓ Final system status:")
        logger.info(f"  - Overall health: {health_status['overall_status']}")
        logger.info(f"  - Total processing time: {final_stats['total_processing_time']:.1f}s")
        logger.info(f"  - Resource usage: Normal")
        logger.info(f"  - Error count: {final_stats['total_errors']}")
        
    except Exception as e:
        logger.warning(f"Final status check limited: {e}")
    
    # 15. Cleanup
    logger.info("15. Cleaning up...")
    
    try:
        await pipeline.shutdown()
        logger.info("✓ Pipeline shutdown completed")
        
    except Exception as e:
        logger.error(f"Shutdown error: {e}")
    
    logger.info("=== Demo completed successfully! ===")
    logger.info("\nKey features demonstrated:")
    logger.info("  ✓ Pipeline configuration and initialization")
    logger.info("  ✓ Resource management and optimization")
    logger.info("  ✓ Processing estimation and planning")
    logger.info("  ✓ Health monitoring and alerting")
    logger.info("  ✓ Checkpointing and recovery")
    logger.info("  ✓ Streaming capabilities")
    logger.info("  ✓ Cloud integration (local mode)")
    logger.info("  ✓ Performance profiling")
    logger.info("  ✓ Error handling and fallbacks")
    logger.info("  ✓ Comprehensive logging and statistics")


if __name__ == "__main__":
    # Run the complete pipeline demonstration
    asyncio.run(main()) 