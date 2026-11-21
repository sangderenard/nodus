# Nodus broadcast media matrix (2026-11-13)

The graph is the producer and authority for media frames. A broadcast exit is
an output row of an independent table; browser viewers are readers, never
producers. The same typed frame edge can feed more than one exit. Transport
selection belongs to the graph's declared configuration, not to a second
renderer or an HTTP request that advances the graph.

| Payload | TCP | UDP |
|---|---|---|
| Video / UI frame | **Current native building block:** `TcpByteStreamTool` broadcast mode accepts one U8 frame and sends the existing u64-length wire frame to each admitted loopback reader. Each reader has an independent current frame and at most one latest pending frame. A slow reader loses stale pending frames; it cannot block the graph or another reader. Meowdoku's current `/api/nodus/frame` HTTP mirror is a temporary reader-facing adapter, not this graph exit. | **Planned:** packetize an explicitly bounded image frame with stream/frame/chunk IDs. A receiver discards any incomplete frame at its deadline. Do not send an oversized image as one UDP datagram or claim delivery from a successful `sendto`. |
| Audio | **Planned:** send independently framed audio chunks through the same TCP broadcast exit, identified by an application envelope. Keep audio timestamps and format with the chunk; drop stale chunks for slow readers. | **Existing:** Radio's `radio.udp_broadcast` and `radio.pcm_rtp` rows use the nonblocking UDP route. RTP/RTCP supplies timestamps and stream end for PCM. `udp_router` remains the service-owned socket path. |

The current TCP byte-stream row is loopback-only and uses the existing
`ContractListener` Hello admission. The new broadcast mode is output-only and
best effort; the old single-peer duplex mode retains its bounded backpressure
and uncertain-delivery behavior. TCP still delivers bytes in order within a
connection. There is no application ACK, replay, or promise that a reader will
see every media frame. A reader that disconnects rejoins at a future frame.

Both media types should eventually be selectable on both transports through
one graph configuration vocabulary: payload type, wire envelope/version,
transport, endpoint, maximum frame size, reader cap, timing and drop policy.
The existing Radio UDP configuration and TCP byte-stream registration are
separate concrete paths today. Do not silently claim that one configuration
already drives both.

An optional later control row may sample a timer window of per-reader TCP
queue age/replaced frames and receiver-reported UDP sequence gaps. UDP send
failures alone are not receiver loss measurements. A throttle may lower frame
rate, size or audio bitrate from those signals; a work distributor may move
new frames to another declared exit at frame boundaries. It must retain the
graph's frame identity and publish the chosen route so readers can switch.
Neither throttle nor route switching is implemented by this change.
