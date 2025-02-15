from __future__ import annotations
from typing import AnyStr, Dict, List, Optional, Tuple
from .dag import DagEdge, get_outgoing_edges, topo_sort
from ._utils import convert_kwargs_to_cmd_line_args
from functools import reduce
import copy
import operator
import subprocess

from ._ffmpeg import input
from ._filters import output
from .nodes import (
    MergeOutputsNode,
    OutputStream,
    get_stream_spec_nodes,
    FilterNode,
    GlobalNode,
    InputNode,
    OutputNode,
    output_operator,
)

from collections.abc import Sequence


class Error(Exception):
    def __init__(self, cmd, stdout, stderr):
        super(Error, self).__init__("{} error (see stderr output for detail)".format(cmd))
        self.stdout = stdout
        self.stderr = stderr


def _get_input_args(input_node: InputNode) -> List[str]:
    if input_node.name == input.__name__:
        kwargs = copy.copy(input_node.kwargs)
        filename = kwargs.pop("filename")
        fmt = kwargs.pop("format", None)
        video_size = kwargs.pop("video_size", None)
        args: List[str] = []

        if fmt:
            args += ["-f", fmt]
        if video_size:
            args += ["-video_size", "{}x{}".format(video_size[0], video_size[1])]

        args += convert_kwargs_to_cmd_line_args(kwargs)
        args += ["-i", filename]

        return args
    raise ValueError(f"Unsupported input node: {input_node}")


def _format_input_stream_name(
    stream_name_map: Dict[Tuple[InputNode, Optional[str]], str],
    edge: DagEdge,
    is_final_arg: bool = False,
) -> str:
    prefix = stream_name_map[edge.upstream_node, edge.upstream_label]  # type: ignore
    if not edge.upstream_selector:
        suffix = ""
    else:
        suffix = f":{edge.upstream_selector}"

    if is_final_arg and isinstance(edge.upstream_node, InputNode):
        # Special case: `-map` args should not have brackets for input nodes.
        return f"{prefix}{suffix}"
    else:
        return f"[{prefix}{suffix}]"


def _format_output_stream_name(stream_name_map: Dict[Tuple[InputNode, Optional[str]], str], edge: DagEdge):
    return f"[{stream_name_map[edge.upstream_node, edge.upstream_label]}]"  # type: ignore


def _get_filter_spec(
    node: FilterNode,
    outgoing_edge_map,
    stream_name_map: Dict[Tuple[InputNode, Optional[str]], str],
):
    incoming_edges = node.incoming_edges
    outgoing_edges = get_outgoing_edges(node, outgoing_edge_map)
    inputs = [_format_input_stream_name(stream_name_map, edge) for edge in incoming_edges]
    outputs = [_format_output_stream_name(stream_name_map, edge) for edge in outgoing_edges]
    filter_spec = "".join(inputs) + node._get_filter(outgoing_edges) + "".join(outputs)
    return filter_spec


def _allocate_filter_stream_names(filter_nodes, outgoing_edge_maps, stream_name_map):
    stream_count = 0
    for upstream_node in filter_nodes:
        outgoing_edge_map = outgoing_edge_maps[upstream_node]
        for upstream_label, downstreams in sorted(outgoing_edge_map.items()):
            if len(downstreams) > 1:
                # TODO: automatically insert `splits` ahead of time via graph transformation.
                raise ValueError(
                    "Encountered {} with multiple outgoing edges with same upstream "
                    "label {!r}; a `split` filter is probably required".format(upstream_node, upstream_label)
                )
            stream_name_map[upstream_node, upstream_label] = "s{}".format(stream_count)
            stream_count += 1


def _get_filter_arg(filter_nodes, outgoing_edge_maps, stream_name_map) -> str:
    _allocate_filter_stream_names(filter_nodes, outgoing_edge_maps, stream_name_map)
    filter_specs = [
        _get_filter_spec(node, outgoing_edge_maps[node], stream_name_map) for node in filter_nodes
    ]
    return ";".join(filter_specs)


def _get_global_args(node: GlobalNode) -> List[str]:
    return list(node.args)


def _get_output_args(
    node: OutputNode,
    stream_name_map: Dict[Tuple[InputNode, Optional[str]], str],
) -> List[str]:
    if node.name != output.__name__:
        raise ValueError("Unsupported output node: {}".format(node))

    args: List[str] = []
    if len(node.incoming_edges) == 0:
        raise ValueError("Output node {} has no mapped streams".format(node))

    for edge in node.incoming_edges:
        # edge = node.incoming_edges[0]
        stream_name = _format_input_stream_name(stream_name_map, edge, is_final_arg=True)
        if stream_name != "0" or len(node.incoming_edges) > 1:
            args += ["-map", stream_name]

    kwargs = copy.copy(node.kwargs)
    filename = kwargs.pop("filename")
    if "format" in kwargs:
        args += ["-f", kwargs.pop("format")]
    if "video_bitrate" in kwargs:
        args += ["-b:v", str(kwargs.pop("video_bitrate"))]
    if "audio_bitrate" in kwargs:
        args += ["-b:a", str(kwargs.pop("audio_bitrate"))]
    if "video_size" in kwargs:
        video_size = kwargs.pop("video_size")
        if not isinstance(video_size, (str, bytes)) and isinstance(video_size, Sequence):
            video_size = "{}x{}".format(video_size[0], video_size[1])
        args += ["-video_size", video_size]
    args += convert_kwargs_to_cmd_line_args(kwargs)
    args += [filename]

    return args


@output_operator()
def get_args(stream_spec: OutputStream, overwrite_output=False) -> List[str]:
    """Build command-line arguments to be passed to ffmpeg."""
    nodes = get_stream_spec_nodes(stream_spec)
    args: List[str] = []
    # TODO: group nodes together, e.g. `-i somefile -r somerate`.
    sorted_nodes, outgoing_edge_maps = topo_sort(nodes)
    input_nodes: List[InputNode] = []
    output_nodes: List[OutputNode] = []
    global_nodes: List[GlobalNode] = []
    filter_nodes: List[FilterNode] = []
    for node in sorted_nodes:
        if isinstance(node, InputNode):
            input_nodes.append(node)
        elif isinstance(node, OutputNode):
            output_nodes.append(node)
        elif isinstance(node, GlobalNode):
            global_nodes.append(node)
        elif isinstance(node, FilterNode):
            filter_nodes.append(node)

    stream_name_map = {(node, None): str(i) for i, node in enumerate(input_nodes)}
    filter_arg = _get_filter_arg(filter_nodes, outgoing_edge_maps, stream_name_map)
    args += reduce(operator.add, [_get_input_args(node) for node in input_nodes])
    if filter_arg:
        args += ["-filter_complex", filter_arg]
    args += reduce(
        operator.add,
        [_get_output_args(node, stream_name_map) for node in output_nodes],  # type: ignore
    )
    args += reduce(operator.add, [_get_global_args(node) for node in global_nodes], [])
    if overwrite_output:
        args += ["-y"]
    return args


@output_operator()
def compile(stream_spec: OutputStream, cmd="ffmpeg", overwrite_output=False) -> List[str]:
    """Build command-line for invoking ffmpeg.

    The :meth:`run` function uses this to build the command line
    arguments and should work in most cases, but calling this function
    directly is useful for debugging or if you need to invoke ffmpeg
    manually for whatever reason.

    This is the same as calling :meth:`get_args` except that it also
    includes the ``ffmpeg`` command as the first argument.
    """
    if isinstance(cmd, (str, bytes)):
        cmd = [cmd]
    elif not isinstance(cmd, list):
        cmd = list(cmd)
    return cmd + get_args(stream_spec, overwrite_output=overwrite_output)


@output_operator()
def run_async(
    stream_spec: OutputStream,
    cmd="ffmpeg",
    pipe_stdin: bool = False,
    pipe_stdout: bool = False,
    pipe_stderr: bool = False,
    quiet: bool = False,
    overwrite_output: bool = False,
    cwd=None,
):
    """Asynchronously invoke ffmpeg for the supplied node graph.

    Args:
        pipe_stdin: if True, connect pipe to subprocess stdin (to be
            used with ``pipe:`` ffmpeg inputs).
        pipe_stdout: if True, connect pipe to subprocess stdout (to be
            used with ``pipe:`` ffmpeg outputs).
        pipe_stderr: if True, connect pipe to subprocess stderr.
        quiet: shorthand for setting ``capture_stdout`` and
            ``capture_stderr``.
        **kwargs: keyword-arguments passed to ``get_args()`` (e.g.
            ``overwrite_output=True``).

    Returns:
        A `subprocess Popen`_ object representing the child process.

    Examples:
        Run and stream input::

            process = (
                ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
                .output(out_filename, pix_fmt='yuv420p')
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )
            process.communicate(input=input_data)

        Run and capture output::

            process = (
                ffmpeg
                .input(in_filename)
                .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run_async(pipe_stdout=True, pipe_stderr=True)
            )
            out, err = process.communicate()

        Process video frame-by-frame using numpy::

            process1 = (
                ffmpeg
                .input(in_filename)
                .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run_async(pipe_stdout=True)
            )

            process2 = (
                ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
                .output(out_filename, pix_fmt='yuv420p')
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )

            while True:
                in_bytes = process1.stdout.read(width * height * 3)
                if not in_bytes:
                    break
                in_frame = (
                    np
                    .frombuffer(in_bytes, np.uint8)
                    .reshape([height, width, 3])
                )
                out_frame = in_frame * 0.3
                process2.stdin.write(
                    frame
                    .astype(np.uint8)
                    .tobytes()
                )

            process2.stdin.close()
            process1.wait()
            process2.wait()

    .. _subprocess Popen: https://docs.python.org/3/library/subprocess.html#popen-objects
    """
    args = compile(stream_spec, cmd, overwrite_output=overwrite_output)
    stdin_stream = subprocess.PIPE if pipe_stdin else None
    stdout_stream = subprocess.PIPE if pipe_stdout else None
    stderr_stream = subprocess.PIPE if pipe_stderr else None
    if quiet:
        stderr_stream = subprocess.STDOUT
        stdout_stream = subprocess.DEVNULL
    return subprocess.Popen(
        args,
        stdin=stdin_stream,
        stdout=stdout_stream,
        stderr=stderr_stream,
        cwd=cwd,
    )


@output_operator()
def run(
    stream_spec: OutputStream,
    cmd="ffmpeg",
    capture_stdout: bool = False,
    capture_stderr: bool = False,
    input: Optional[AnyStr] = None,
    quiet: bool = False,
    overwrite_output: bool = False,
    cwd=None,
):
    """Invoke ffmpeg for the supplied node graph.

    Args:
        capture_stdout: if True, capture stdout (to be used with
            ``pipe:`` ffmpeg outputs).
        capture_stderr: if True, capture stderr.
        quiet: shorthand for setting ``capture_stdout`` and ``capture_stderr``.
        input: text to be sent to stdin (to be used with ``pipe:``
            ffmpeg inputs)
        **kwargs: keyword-arguments passed to ``get_args()`` (e.g.
            ``overwrite_output=True``).

    Returns: (out, err) tuple containing captured stdout and stderr data.
    """
    process = run_async(
        stream_spec,
        cmd,
        pipe_stdin=input is not None,
        pipe_stdout=capture_stdout,
        pipe_stderr=capture_stderr,
        quiet=quiet,
        overwrite_output=overwrite_output,
        cwd=cwd,
    )
    out, err = process.communicate(input.encode() if isinstance(input, str) else input)
    retcode = process.poll()
    if retcode:
        raise Error("ffmpeg", out, err)
    return out, err


@output_operator()
def global_args(stream: OutputStream, *args) -> OutputStream:
    """Add extra global command-line argument(s), e.g. ``-progress``."""
    return GlobalNode(stream, global_args.__name__, args).stream()


@output_operator()
def overwrite_output(stream: OutputStream) -> OutputStream:
    """Overwrite output files without asking (ffmpeg ``-y`` option)

    Official documentation: `Main options <https://ffmpeg.org/ffmpeg.html#Main-options>`__
    """
    return GlobalNode(stream, overwrite_output.__name__, ["-y"]).stream()


@output_operator()
def merge_outputs(*streams: OutputStream) -> OutputStream:
    """Include all given outputs in one ffmpeg command line"""
    return MergeOutputsNode(streams, merge_outputs.__name__).stream()


class OutputOperators:
    get_args = get_args
    compile = compile
    run_async = run_async
    run = run
    global_args = global_args
    overwrite_output = overwrite_output
    merge_outputs = merge_outputs


__all__ = [
    "Error",
    "get_args",
    "compile",
    "run_async",
    "run",
    "global_args",
    "overwrite_output",
    "merge_outputs",
]
