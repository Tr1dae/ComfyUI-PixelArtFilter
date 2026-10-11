from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .nodes import PixelArtFilterAutoRefiner, PixelArtFilterGridCollapse, PixelArtFilterOutlineExpansion, PixelArtFilterPaletteReduce, PixelArtFilterPixelize, PixelArtFilterPoseCompositor, PixelArtFilterPreResize, PixelArtFilterRefiner, PixelArtFilterSimple, PixelArtFilterTemporal, PixelArtFilterTwoPass


class PixelArtFilterExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [PixelArtFilterTwoPass, PixelArtFilterTemporal, PixelArtFilterRefiner, PixelArtFilterAutoRefiner, PixelArtFilterSimple, PixelArtFilterPixelize, PixelArtFilterOutlineExpansion, PixelArtFilterPreResize, PixelArtFilterPoseCompositor, PixelArtFilterPaletteReduce, PixelArtFilterGridCollapse]


async def comfy_entrypoint() -> PixelArtFilterExtension:
    return PixelArtFilterExtension()
