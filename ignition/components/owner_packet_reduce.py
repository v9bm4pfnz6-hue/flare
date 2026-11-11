# SPDX-License-Identifier: Apache-2.0

"""Bounded FP32 DMA reduction retaining the full tensor's ring-owner order."""

import torch
from b12x.comm.pcie import DmaAllReduce

from .collectives import PreparedAllReduce


def aligned_extent(elements, world):
    """Elements per rank once a flat tensor holds complete 32-byte FP32 groups.

    The ring addresses every rank's shard as whole eight-element FP32 groups, so a payload whose
    element count is not a multiple of `world * 8` is padded at the tail before reduction.
    """

    if type(elements) is not int or elements <= 0 or type(world) is not int or world <= 0:
        raise ValueError("aligned extents require positive element and rank counts")
    groups = world * 8
    return -(-elements // groups) * 8


def packet_intervals(elements, world, workspace_elements, max_bytes):
    """Ring-aligned element intervals one prepared owner channel reduces.

    The intervals tile the ring-aligned extent exactly once. A workspace sized
    to the aligned extent keeps an unaligned payload (a 128-row prefill graph)
    inside one ring invocation instead of a full invocation for the tail
    remainder.
    """

    extent = aligned_extent(elements, world)
    capacity = min(int(workspace_elements), int(max_bytes) // 4) // world // 8 * 8
    if capacity <= 0:
        raise ValueError(
            "owner-packet scratch cannot hold one aligned element group per rank"
        )
    intervals = []
    for first in range(0, extent, capacity):
        end = min(first + capacity, extent)
        intervals.append((first, end))
    return extent, tuple(intervals)


def copy_engine_reduction(source, output, *, runtime, workspace, staging, session, coordinator, name):
    """The prepared reduction of a copy-engine channel for one BF16 matrix.

    A tree channel reduces BF16 directly; a ring channel reduces FP32 owner packets in `workspace`.
    """
    if getattr(runtime, "tree", None) is not None:
        return PreparedTreeAllReduce(source, output, staging=staging, runtime=runtime, session=session,
                                     coordinator=coordinator, name=name)
    return PreparedOwnerPacketAllReduce(source, output, workspace=workspace, staging=staging, runtime=runtime,
                                        session=session, coordinator=coordinator, name=name)


class PreparedTreeAllReduce:
    """BF16 reduction routed over the ranks' PCIe switch tree on the copy engines.

    The channel sums BF16 values in FP32 and rounds once at each chunk's owner, so this component
    adds no promotion and no rounding of its own. A payload whose element count is not a multiple
    of `world * 8` is reduced in caller-owned padded staging; the tail beyond the payload is
    reduced with it and never read. Input and output may be exact aliases or disjoint. Calls
    sharing the channel serialize.
    """

    def __init__(self, source, output, *, runtime, session, coordinator, name, staging=None):
        if not isinstance(runtime, DmaAllReduce) or getattr(runtime, "tree", None) is None:
            raise ValueError("a tree reduction requires a copy-engine channel prepared with a tree topology")
        if (source.ndim != 2 or source.numel() == 0 or source.dtype != torch.bfloat16
                or output.shape != source.shape or output.dtype != source.dtype
                or source.device != output.device or not source.is_contiguous() or not output.is_contiguous()):
            raise ValueError("a tree reduction requires matching contiguous BF16 matrices")
        world, elements = runtime.world_size, source.numel()
        extent = aligned_extent(elements, world)
        self.output, self.elements, self.padded = output, elements, None
        if extent * world == elements:
            self.reduce = PreparedAllReduce(source, output, runtime=runtime, session=session,
                                            coordinator=coordinator, name=name + ".tree")
            return
        if (staging is None or staging.device != source.device or staging.dtype != source.dtype
                or not staging.is_contiguous() or staging.numel() < extent * world):
            raise ValueError("unaligned tree reductions require caller-owned padded BF16 staging")
        # Staging is shared by graphs of several shapes, so the tail may hold what an earlier
        # reduction left there. Every rank holds the same values, and only the payload is copied out.
        self.padded = staging[: extent * world]
        packet = self.padded.view(world, extent)
        self.reduce = PreparedAllReduce(packet, packet, runtime=runtime, session=session,
                                        coordinator=coordinator, name=name + ".tree")
        self.packet = packet

    def __call__(self, source):
        if self.padded is None:
            return self.reduce(source)
        self.padded[: self.elements].copy_(source.view(-1))
        self.reduce(self.packet)
        self.output.view(-1).copy_(self.padded[: self.elements])
        return self.output


class PreparedOwnerPacketAllReduce:
    """Promote, reduce in place, and round once without a full-sized FP32 copy.

    Each packet takes the same interval from every original owner shard.
    Reducing consecutive token chunks instead would change the ring's FP32
    addition order. Input/output may be exact aliases or disjoint; scratch is
    caller-owned and disjoint from both. Calls sharing the channel serialize.
    """

    def __init__(
        self, source, output, *, workspace, runtime, session, coordinator, name, staging=None
    ):
        if not isinstance(runtime, DmaAllReduce):
            raise ValueError("owner-packet reduction requires a prepared DMA ring")
        # A ring prepared with bf16_gather=True returns BF16-representable FP32 values: each
        # fully reduced owner chunk was rounded once with the conversion this component's final
        # store performs and forwarded as those BF16 bits. The BF16 output is therefore
        # bit-identical to the lossless FP32 ring, so this component needs no wire-specific
        # branch; its final store above must remain the only rounding it applies.
        if (
            source.ndim != 2
            or source.numel() == 0
            or source.dtype != torch.bfloat16
            or output.shape != source.shape
            or output.dtype != source.dtype
            or source.device != output.device
            or not source.is_contiguous()
            or not output.is_contiguous()
        ):
            raise ValueError(
                "owner-packet reduction requires matching contiguous BF16 matrices"
            )
        if (
            workspace.device != source.device
            or workspace.dtype != torch.float32
            or not workspace.is_contiguous()
        ):
            raise ValueError(
                "owner-packet reduction requires caller-owned contiguous FP32 scratch"
            )

        def interval(tensor):
            return tensor.data_ptr(), tensor.data_ptr() + tensor.nbytes

        def overlaps(a, b):
            return a[0] < b[1] and b[0] < a[1]

        source_range, output_range, scratch_range = map(
            interval, (source, output, workspace)
        )
        if (
            overlaps(scratch_range, source_range)
            or overlaps(scratch_range, output_range)
            or source_range != output_range
            and overlaps(source_range, output_range)
        ):
            raise ValueError(
                "owner-packet scratch and partial input/output overlaps are not supported"
            )
        self.world = runtime.world_size
        elements = source.numel()
        self.extent = aligned_extent(elements, self.world)
        self.padded = self.extent * self.world
        if self.padded == elements:
            if staging is not None:
                raise ValueError(
                    "aligned owner-packet reductions do not take padded staging"
                )
            self.padded = None
        else:
            if (
                staging is None
                or staging.device != source.device
                or staging.dtype != source.dtype
                or not staging.is_contiguous()
                or staging.numel() < self.extent * self.world
            ):
                raise ValueError(
                    "unaligned owner-packet reductions require caller-owned padded BF16 staging"
                )
            # Padding rows stay zero: every rank contributes zero there, so the ring rewrites zeros.
            self.padded = staging[: self.extent * self.world]
        self.output, self.parts = output, []
        self.workspace = workspace
        _, intervals = packet_intervals(
            elements, self.world, workspace.numel(), runtime.max_bytes
        )
        for first, end in intervals:
            packet = workspace.view(-1)[: self.world * (end - first)].view(
                self.world, end - first
            )
            reduce = PreparedAllReduce(
                packet,
                packet,
                runtime=runtime,
                session=session,
                coordinator=coordinator,
                name=f"{name}.owner_packet{len(self.parts)}",
                # The packet is promoted from BF16 activations, exactly.
                bf16_values=True,
            )
            self.parts.append((first, end, packet, reduce))

    def __call__(self, source):
        if self.padded is None:
            source = source.view(self.world, self.extent)
            output = self.output.view(self.world, self.extent)
        else:
            self.padded[: source.numel()].copy_(source.view(-1))
            source = output = self.padded.view(self.world, self.extent)
        for first, end, packet, reduce in self.parts:
            packet.copy_(source[:, first:end])
            reduce(packet)
            output[:, first:end].copy_(packet)
        if self.padded is not None:
            self.output.view(-1).copy_(self.padded[: self.output.numel()])
        return self.output
