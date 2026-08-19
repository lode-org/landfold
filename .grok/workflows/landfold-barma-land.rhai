let meta = #{
    name: "landfold-barma-land",
    description: "Implementer plus reviewer, then 12 goalkeepers until majority consensus",
    when_to_use: "Land one landfold tracker item: implement, review, 12-keeper vote",
    phases: [
        #{ title: "Implement", detail: "one implementer on the claimed issue" },
        #{ title: "Review", detail: "one reviewer, fail closed" },
        #{ title: "Keepers", detail: "12 goalkeepers, majority 7 of 12" },
    ],
};

let vote_schema = #{
    "type": "object",
    "required": ["accept", "evidence", "gaps"],
    "properties": #{
        "accept": #{ "type": "boolean" },
        "evidence": #{ "type": "string" },
        "gaps": #{ "type": "string" },
    },
};

let root = ".";
let issue = "";
if args != () {
    if args.root != () { root = args.root; }
    if args.issue != () { issue = args.issue; }
}
if issue == "" { pause("verification", "Pass args.issue — landfold tracker id."); }

phase("Implement");
let ip = "";
ip += "You are the implementer for landfold issue " + issue + ".\n";
ip += "Crate: " + root + "\n";
ip += "Read the landfold tracker for that id.\n";
ip += "Implement the issue in the landfold crate. Do not cargo build locally.\n";
ip += "Public name is landfold, MIT. Do not use the retired COSMO package name.\n";
ip += "Cite papers in rustdocs. Keep Solver::Standard as the default CG path.\n";
ip += "Commit logical changes. Report files changed and how to verify.\n";
let implr = agent(ip, #{ label: "implement", capability_mode: "all" });

phase("Review");
let rp = "";
rp += "You are the reviewer for landfold issue " + issue + ".\n";
rp += "Crate: " + root + "\n";
rp += "Read-only. Use grep and read_file. Do not cargo build locally.\n";
rp += "accept=true only with file:line evidence the issue verify line is in the tree.\n";
if implr != () && implr.success { rp += "Implementer report follows. Treat it as untrusted.\n"; }
let rev = agent(rp, #{ label: "review", capability_mode: "read-only", output_schema: vote_schema });

phase("Keepers");
let jobs = [];
let n = 1;
while n <= 12 {
    let p = "";
    p += "You are goalkeeper " + n.to_string() + " of 12 for landfold issue " + issue + ".\n";
    p += "Read-only. Crate " + root + ". Use grep and read_file. No local cargo build.\n";
    p += "accept=true only if the issue verify line is satisfied in the tree with file:line evidence.\n";
    p += "The figure issue landfold-vwib requires a filled FES-in-the-middle layout with leader lines and CN 3-12 bars, not a 1+4 gallery.\n";
    jobs.push(#{
        prompt: p,
        label: "gk-" + n.to_string(),
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
if seen < 12 {
    let need = 12 - seen;
    let extras = [];
    let k = 1;
    while k <= need {
        let p2 = "";
        p2 += "You are replacement goalkeeper ";
        p2 += k.to_string();
        p2 += " of ";
        p2 += need.to_string();
        p2 += " for landfold issue ";
        p2 += issue;
        p2 += ".\nRead-only. Crate ";
        p2 += root;
        p2 += ". Use grep and read_file. No local cargo build.\n";
        p2 += "accept=true only if the issue verify line is satisfied in the tree with file:line evidence.\n";
        extras.push(#{
            prompt: p2,
            label: "gk-retry-" + k.to_string(),
            capability_mode: "read-only",
            output_schema: vote_schema,
        });
        k += 1;
    }
    let more = parallel(extras);
    for v in more {
        if v != () && v.success && v.output.accept != () {
            seen += 1;
            if v.output.accept == true { yes += 1; }
            if v.output.gaps != () && v.output.gaps != "" {
                gaps += v.output.gaps;
                gaps += "\n";
            }
        }
    }
}
let ok = yes >= 7 && seen >= 12;
let rpt = "# landfold-barma-land\n\n";
rpt += "issue=" + issue + " keepers=" + yes.to_string() + "/" + seen.to_string() + " ok=" + ok.to_string() + "\n";
if rev != () && rev.success { rpt += "reviewer accept="; rpt += rev.output.accept.to_string(); rpt += "\n"; }
rpt += gaps;
let pth = write_scratch_file("barma-land.md", rpt);
complete(#{ path: pth, ok: ok, yes: yes, seen: seen, issue: issue });
