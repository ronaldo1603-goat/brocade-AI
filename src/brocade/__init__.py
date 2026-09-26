"""brocade - inference package for the CHINTEXDB-PERU28 (Chinchero textile) models.

    from brocade import load_detector, draw_detections
    det = load_detector("weights/miniyolo_checkpoint.pt")      # or an Ultralytics best.pt
    result = det("photo.jpg")
    draw_detections("photo.jpg", result).save("photo_det.jpg")
"""

__version__ = "0.1.0"