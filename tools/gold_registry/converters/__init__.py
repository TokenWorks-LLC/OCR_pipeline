from .alto_xml_converter import AltoXmlConverter
from .coco_converter import CocoConverter
from .json_boxes_converter import JsonBoxesConverter
from .line_pairs_converter import LinePairsConverter
from .local_gold_converter import (
    LOCAL_GOLD_SUPPORTED_FORMATS,
    LocalGoldConverter,
    build_source_page_key,
    canonical_lookup_key,
    extract_local_gold_text,
)
from .page_xml_converter import PageXmlConverter

ALL_CONVERTERS = [
    PageXmlConverter(),
    AltoXmlConverter(),
    CocoConverter(),
    JsonBoxesConverter(),
    LinePairsConverter(),
]
