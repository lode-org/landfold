let meta = #{
    name: "landfold-barma-quality",
    description: "12-voter Borda plus majority quality raft for the landfold crate",
    when_to_use: "After a landfold kernel lands, before calling the step done",
    phases: [
        #{ title: "Quality", detail: "12 voters on idiomatic Rust and citations" },
    ],
};

let vote_schema = #{
    "type": "object",
    "required": ["accept", "evidence", "rank1", "rank2", "rank3", "gaps"],
    "properties": #{
        "accept": #{ "type": "boolean" },
        "evidence": #{ "type": "string" },
        "rank1": #{ "type": "string" },
        "rank2": #{ "type": "string" },
        "rank3": #{ "type": "string" },
        "gaps": #{ "type": "string" },
    },
};

let root = ".";
if args != () {
    if args.root != () { root = args.root; }
}

phase("Quality");
let jobs = [];
let n = 1;
while n <= 12 {
    let p = "";
    p += "You are BARMA voter " + n.to_string() + " of 12 for landfold quality.\n";
    p += "Read-only. Use read_file and grep. Do not cargo build locally.\n";
    p += "Crate root: " + root + "\n";
    p += "Read src/lib.rs src/transfer.rs src/mds.rs src/stress.rs src/cg.rs src/search.rs src/iter.rs src/bin/landfold.rs Cargo.toml LICENSE.\n";
    p += "Checks: MIT license; crate name landfold; rustdocs cite PNAS 2011 / Torgerson / Polak-Ribiere / Robbins-Monro on the kernels they implement; Solver::Standard is the default full-pair CG; no GPL; idiomatic Rust (no C++ type names).\n";
    p += "accept=true only with file:line evidence. rank1/rank2/rank3 = Borda 3/2/1 improvements.\n";
    jobs.push(#{
        prompt: p,
        label: "q-" + n.to_string(),
        capability_mode: "read-only",
        output_schema: vote_schema,
    });
    n += 1;
}
let votes = parallel(jobs);
let yes = 0;
let seen = 0;
let gaps = "";
for v in votes {
    if v != () && v.success && v.output.accept != () {
        seen += 1;
        if v.output.accept == true { yes += 1; }
        if v.output.gaps != () && v.output.gaps != "" {
            gaps += v.output.gaps;
            gaps += "\n";
        }
    }
}
let committed = yes >= 7 && seen >= 12;
let rpt = "# landfold BARMA quality\n\n";
rpt += "ok=" + committed.to_string() + " yes=" + yes.to_string() + "/" + seen.to_string() + "\n";
rpt += gaps;
let pth = write_scratch_file("barma-quality.md", rpt);
complete(#{ path: pth, ok: committed, yes: yes, seen: seen });
