/* UVLED gain calibration
 * measure.d
*/

module measure;

import fadcdacs, skynet, packet;

enum MAX_MIR = 64;
enum MAX_CHN = 320;

struct Pulse {
    double pedestal, area;
}

struct Trigger {
    Waveform[] waveforms;
    ushort id, hits;
}

class MeasureException : Exception
{
    this(string msg) {
        super(msg);
    }
}

double filteredMean(const ubyte[] data)
{
    import core.stdc.stdlib : alloca;
    import std.algorithm.iteration : filter, mean;
    import std.algorithm.sorting : sort;
    import std.math : abs;
    auto n = data.length;
    auto dd = (cast(ubyte*)alloca(n))[0 .. n];
    dd[] = data[];
    dd.sort;
    auto median = (dd[(n - 1) / 2] + dd[n / 2] + 1) / 2;
    auto maxerr = 3 * (dd[3 * n / 4] - dd[n / 4]); // ~ 4 sigma
    return dd.filter!(v => abs(v - median) <= maxerr).mean;
}

import std.traits : isFloatingPoint;
import std.range.primitives : isInputRange, isInfinite, ElementType;
auto filteredMean(R)(R data)
    if (isInputRange!R && !isInfinite!R && isFloatingPoint!(ElementType!R))
{
    import std.algorithm.iteration : filter, mean;
    import std.algorithm.sorting : sort;
    import std.array : array;
    import std.math : isNaN, abs;
    auto dd = data.filter!(v => !v.isNaN).array.sort;
    auto n = dd.length;
    if (n <= 0) // || n < data.length / 2)
        return double.nan;
    auto median = 0.5 * (dd[(n - 1) / 2] + dd[n / 2]);
    auto maxerr = 2.965 * (dd[3 * n / 4] - dd[n / 4]); // ~ 4 sigma
    return dd.filter!(v => abs(v - median) <= maxerr).mean;
}

Pulse pulseArea(ubyte[] wave)
{
    import std.algorithm.iteration : sum;
    import std.algorithm.searching : maxIndex;
    import std.range : iota;
    // start with estimated pedestal for first 16 samples
    int estPed = (wave[0 .. 16].sum + 8) / 16;
    // find waveform peak
    size_t peak = 16 + wave[16 .. $-16].maxIndex;
    // find index of rising and falling edges of pulse
    int hmax = (estPed + wave[peak]) / 2;
    size_t rise = peak, fall = peak;
    while (rise > 16 && (wave[rise] > hmax || wave[rise - 1] > hmax)) --rise;
    while (fall < wave.length - 16 && (wave[fall] > hmax || wave[fall + 1] > hmax)) ++fall;
    if (rise == 16 || fall == wave.length - 16)  // can't do anything with this
        return Pulse(double.nan, double.nan);
    auto width = fall - rise;
    // re-compute pedestal, pulse undershoot, pulse area
    auto pedestal = filteredMean(wave[0 .. rise - 5]);
    if (wave[peak] == 255) // pulse saturated, area invalid
        return Pulse(pedestal, double.nan);
    double area;
    if (width > 8 && width < 12) { // width should be 1uS * 1sample / 100nS = 10samples
        auto under = pedestal - filteredMean(wave[fall + 5 .. $]);
        // area under pulse from 5 samples before rising edge to 5 samples after falling edge
        // and approximating undershoot as linear drop for duration of pulse
        area = cast(double)(wave[rise - 5 .. fall + 5].sum)
            - pedestal * (width + 10) + 0.5 * under * width;
    }
    return Pulse(pedestal, area);
}

void measureEvents(string[string] config, Dacs[][] dacs, ref Skynet sky,
                   Pulse[][][] events, string triggers)
{
    import std.stdio : writeln;
    import std.format : format;
    import std.conv : to;
    import std.math : abs;
    import core.time : MonoTime, seconds;
    import std.regex : ctRegex, matchFirst;
    import std.experimental.allocator.building_blocks.free_list : FreeList;
    import std.experimental.allocator.building_blocks.region : InSituRegion;
    import packet;
    // script notice messages
    static immutable script_complete = ctRegex!(r"'([^']+)' complete");
    static immutable script_terminated = ctRegex!(r"'([^']+)' terminated");
    // 10 mirrors need: (10 mir * 320 pkt/mir + 1 pkt) * 128b/pkt = 409728b
    const packetSize = 64 * ushort.sizeof;
    FreeList!(InSituRegion!(512 * 1024), packetSize) packetBufs;
    Trigger[MAX_MIR] trig;
    auto buf = cast(ushort[])(packetBufs.allocate(packetSize));
    ushort curMir;
    // configure DACs and start uvled_balance script
    foreach (mir, mdac; dacs) {
        if (!mdac) continue;
        foreach (pkt; ConfigDacs(buf, mdac)) {
            pkt[1] |= cast(ushort)(mir << 8);
            sky.sock.send(pkt);
        }
    }
    with (PacketTypes)
        sky.needPkts(NoticePkt, FADCDataPkt, EndEventPkt);
    auto script = config["script"];
    sky.command(script, " ", config[triggers]); // script will load DACs "dac base"
    // Data should start to come in within a few seconds, then continue at atleast 1Hz rate
    auto timeout = MonoTime.currTime + seconds(30);
    data_loop: while (true) {
        if (sky.receive(buf, timeout - MonoTime.currTime) <= 0)
            throw new MeasureException("Failed to read FADC data");
        auto pkt = Packet(buf);
        with (PacketTypes) switch (pkt.pktType) {
          case NoticePkt:
            auto note = Notice(pkt);
            if (note.noteType != NoticeTypes.NoticeScript) // not a script notice packet
                break;
            auto match = note.text.matchFirst(script_complete);
            if (match && match[1] == script) // script finished
                break data_loop;
            match = note.text.matchFirst(script_terminated);
            if (match && match[1] == script) // something went wrong...
                throw new MeasureException("Failed to run skynet script: " ~ script);
            break;
          case FADCDataPkt:
            auto fadc = FADCData(pkt);
            if (fadc.toe == 0x39) { // waveform
                auto waveform = Waveform(fadc);
                if (trig[curMir].id != waveform.trigId) {
                    // Wrong TID! Scan all mirrors for this TID
                    ushort m = 1;
                    while (m < MAX_MIR && trig[m].id != waveform.trigId) ++m;
                    if (m == MAX_MIR) {
                        writeln(format!"Waveform TID %04x does not match any active event!"(waveform.trigId));
                        break;
                    }
                    curMir = m;
                }
                trig[curMir].waveforms ~= waveform;
                if (waveform.last) curMir = 0;
                // buf is waveform so swap to new read buffer
                buf = cast(ushort[])(packetBufs.allocate(packetSize));
            }
            else if (!fadc.last) { // scan-hit, record TID
                auto scanHit = ScanHit(fadc);
                if (scanHit.trigId % 256 != 0) {
                    curMir = scanHit.mirror;
                    Trigger *mtrg = &trig[curMir];
                    if (!mtrg.id) {
                        mtrg.id = scanHit.trigId;
                        mtrg.waveforms.reserve(MAX_CHN);
                    }
                    mtrg.hits += scanHit.hitTimes.length;
                }
                // else snapshot
            }
            break;
          case EndEventPkt:
            auto endEvent = EndEvent(pkt);
            auto mir = endEvent.mirId;
            auto event = new Pulse[320];
            foreach (waveform; trig[mir].waveforms) {
                event[waveform.tube] = pulseArea(waveform.fadc);
                packetBufs.deallocate(waveform);
            }
            events[mir] ~= event;
            trig[mir].waveforms.length = 0;
            trig[mir].waveforms.assumeSafeAppend;
            trig[mir].id = 0;
            trig[mir].hits = 0;
            timeout = MonoTime.currTime + seconds(10);
            break;
          default:
            break;
        }
    }
}

void measureNpe(string[string] config, Dacs[][] dacs, ref Skynet sky, double[] npe, uint[] trgs)
{
    import std.stdio : write;
    import std.algorithm.iteration : map;
    import std.math : isNaN;
    Pulse[][][MAX_MIR] events;
    measureEvents(config, dacs, sky, events, "trigs-npe");
    foreach (m, mevts; events) {
        if (!mevts) continue; // no events for this mirror
        // normalize areas over events for each tube (remove UVLED pulse-to-pulse variation)
        foreach (t; 0 .. 256) {
            auto mean = mevts.map!(ev => ev[t].area).filteredMean;
            if (!mean.isNaN) {
                foreach (ev; mevts)
                    ev[t].area /= mean;
            }
        }
        // normalize areas over tubes for each event (remove tube-to-tube response variation)
        foreach (ev; mevts) {
            auto mean = ev[0 .. 256].map!(p => p.area).filteredMean;
            if (!mean.isNaN) { // drop bad events...
                foreach (t; 0 .. 256)
                    ev[t].area /= mean;
                trgs[m] += 1;
            }
        }
        // compute mean, variance over events for each tube, mean npe over all tubes
        double npe_mean = 0.0, npe_n = 0.0;
        foreach (t; 0 .. 256) {
            double mean = 0.0, var = 0.0, n = 0.0;
            foreach (ev; mevts) {
                auto area = ev[t].area;
                if (!area.isNaN) {
                    mean += area;
                    var += area * area;
                    n += 1.0;
                }
            }
            if (n < trgs[m] / 2) continue; // dead or intermittant tube
            var -= mean * mean / n;
            mean /= n;
            var /= (n - 1.0);
            npe_mean += mean * mean / var;
            npe_n += 1.0;
        }
        if (npe_n > 0.5)
            npe[m] = 1.548 * npe_mean / npe_n; // including excess noise factor
    }
}
