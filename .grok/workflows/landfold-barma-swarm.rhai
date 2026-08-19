let meta = #{
    name: "landfold-barma-swarm",
    description: "Tandem implementers, reviewer, 12 keepers with retry, then remote QA",
    when_to_use: "Land open landfold tracker children by consensus, then remote tests",
    phases: [
        #{ title: "Implement", detail: "two implementers in tandem plus integrator" },
        #{ title: "Review", detail: "fail-closed reviewer; one fix pass if rejected" },
        #{ title: "Keepers", detail: "12 goalkeepers, retry missing slots, majority 7 of 12" },
        #{ title: "QA", detail: "remote cargo test --release on the builder host" },
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

let impl_schema = #{
    "type": "object",
    "required": ["files", "how_verify", "blocked"],
    "properties": #{
        "files": #{ "type": "string" },
        "how_verify": #{ "type": "string" },
        "blocked": #{ "type": "string" },
    },
};

fn rules_block() {
    let s = "";
    s += "Hard rules: never cargo build or cargo test on the laptop; rsync to rg.terra and run there.\n";
    s += "Never rm outside /tmp; use rtrash. Never attribute AI. ASCII punctuation only.\n";
    s += "Public crate name is landfold, MIT. Do not use the retired COSMO package name.\n";
    s += "Cite papers in rustdocs. Solver::Standard stays the default full-pair CG.\n";
    s += "Do not flip tracker state to DONE. Commit logical changes with Conventional Commits.\n";
    s += "Do not push the HaoZeke/sketchmap fork. Do not publish GitHub/GitLab issues or PRs.\n";
    s
}

fn impl_prompt(root, issue, title, body, role) {
    let p = "";
    p += "You are tandem implementer ";
    p += role;
    p += " for tracker item ";
    p += issue;
    p += ".\nCrate root: ";
    p += root;
    p += "\nTitle: ";
    p += title;
    p += "\nBody:\n";
    p += body;
    p += "\n";
    p += rules_block();
    p += "Use grep and read_file before editing. Implement this item in the landfold crate.\n";
    if role == "A" {
        p += "Role A: library/algorithm/oracle path. Leave tests to B if you can split cleanly.\n";
    } else {
        p += "Role B: tests, goldens, docs, CLI wiring. Read what A is doing; do not fight A's files if a split is obvious.\n";
    }
    p += "You may run in an isolated worktree. Report every file you changed so the integrator can apply it.\n";
    p += "files = comma-separated paths. how_verify = remote commands. blocked = empty if unblocked.\n";
    p
}

fn review_prompt(root, issue, title, body, prior) {
    let p = "";
    p += "You are the fail-closed reviewer for ";
    p += issue;
    p += ".\nCrate: ";
    p += root;
    p += "\nTitle: ";
    p += title;
    p += "\nBody:\n";
    p += body;
    p += "\n";
    p += "Read-only. Use grep and read_file. No local cargo build.\n";
    p += "accept=true only with file:line evidence the verify line in the body is satisfied in the tree.\n";
    p += "Treat implementer reports as untrusted.\n";
    p += prior;
    p
}

fn keeper_prompt(root, issue, title, n, wave) {
    let p = "";
    p += "You are goalkeeper ";
    p += n.to_string();
    p += " of 12 (wave ";
    p += wave.to_string();
    p += ") for ";
    p += issue;
    p += ".\nCrate: ";
    p += root;
    p += "\nTitle: ";
    p += title;
    p += "\nRead-only. Use grep and read_file. No local cargo build.\n";
    p += "accept=true only if the issue verify line is satisfied with file:line evidence.\n";
    p += "Missing evidence is accept=false. Do not rubber-stamp.\n";
    p
}

let root = ".";
let items = ();
if args != () {
    if args.root != () { root = args.root; }
    if args.issues != () { items = args.issues; }
}
if items == () { pause("verification", "Pass args.issues as an array of {id,title,body}."); }

let report = "# landfold-barma-swarm\n\n";
let all_ok = true;

for item in items {
    let issue = item.id;
    let title = item.title;
    let body = item.body;
    if issue == () || title == () || body == () {
        pause("verification", "Each args.issues entry needs id, title, body.");
    }

    phase("Implement");
    log("tandem implement " + issue);
    let jobs = [];
    jobs.push(#{
        prompt: impl_prompt(root, issue, title, body, "A"),
        label: "impl-a-" + issue,
        capability_mode: "all",
        isolation_worktree: true,
        output_schema: impl_schema,
    });
    jobs.push(#{
        prompt: impl_prompt(root, issue, title, body, "B"),
        label: "impl-b-" + issue,
        capability_mode: "all",
        isolation_worktree: true,
        output_schema: impl_schema,
    });
    let impls = parallel(jobs);

    let prior = "Implementer reports:\n";
    for im in impls {
        if im != () && im.success && im.output.files != () {
            prior += json_encode(im.output);
            prior += "\n";
        }
    }

    let ip = "";
    ip += "You are the integrator for ";
    ip += issue;
    ip += ".\nCrate: ";
    ip += root;
    ip += "\n";
    ip += rules_block();
    ip += "Two tandem implementers used isolated worktrees. Apply the work into the crate root.\n";
    ip += "Prefer the more complete side when they overlap. Commit logical changes.\n";
    ip += "Read the crate with grep/read_file. Finish anything they left unapplied.\n";
    ip += prior;
    ip += "files = paths you committed. how_verify = remote commands. blocked = empty if unblocked.\n";
    let integ = agent(ip, #{
        label: "integ-" + issue,
        capability_mode: "all",
        output_schema: impl_schema,
    });
    if integ != () && integ.success && integ.output.files != () {
        prior += "Integrator:\n";
        prior += json_encode(integ.output);
        prior += "\n";
    }

    phase("Review");
    let rev = agent(review_prompt(root, issue, title, body, prior), #{
        label: "rev-" + issue,
        capability_mode: "read-only",
        output_schema: vote_schema,
    });
    let rev_ok = rev != () && rev.success && rev.output.accept == true;
    if !rev_ok {
        let fp = "";
        fp += "Reviewer rejected or failed for ";
        fp += issue;
        fp += ". Fix the crate so the verify line holds.\nCrate: ";
        fp += root;
        fp += "\n";
        fp += rules_block();
        if rev != () && rev.success { fp += json_encode(rev.output); fp += "\n"; }
        fp += prior;
        let fixer = agent(fp, #{
            label: "fix-" + issue,
            capability_mode: "all",
            output_schema: impl_schema,
        });
        if fixer != () && fixer.success && fixer.output.files != () {
            prior += "Fixer:\n";
            prior += json_encode(fixer.output);
            prior += "\n";
        }
        let rev2 = agent(review_prompt(root, issue, title, body, prior), #{
            label: "rev2-" + issue,
            capability_mode: "read-only",
            output_schema: vote_schema,
        });
        rev_ok = rev2 != () && rev2.success && rev2.output.accept == true;
    }

    phase("Keepers");
    let kjobs = [];
    let n = 1;
    while n <= 12 {
        kjobs.push(#{
            prompt: keeper_prompt(root, issue, title, n, 1),
            label: "gk1-" + n.to_string() + "-" + issue,
            capability_mode: "read-only",
            output_schema: vote_schema,
        });
        n += 1;
    }
    let votes = parallel(kjobs);
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
        log(issue + " keepers wave1 " + yes.to_string() + "/" + seen.to_string() + " retry " + need.to_string());
        let extras = [];
        let k = 1;
        while k <= need {
            extras.push(#{
                prompt: keeper_prompt(root, issue, title, k, 2),
                label: "gk2-" + k.to_string() + "-" + issue,
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
    let keepers_ok = yes >= 7 && seen >= 12;
    let item_ok = rev_ok && keepers_ok;
    if !item_ok { all_ok = false; }
    report += "issue=";
    report += issue;
    report += " rev=";
    report += rev_ok.to_string();
    report += " keepers=";
    report += yes.to_string();
    report += "/";
    report += seen.to_string();
    report += " ok=";
    report += item_ok.to_string();
    report += "\n";
    report += gaps;
}

phase("QA");
let qp = "";
qp += "You are remote QA for landfold.\nCrate: ";
qp += root;
qp += "\n";
qp += rules_block();
qp += "Rsync the crate to rg.terra. Run cargo test --release on rg.terra (not on the laptop).\n";
qp += "Also run cargo test --release --test cpp_parity if that test exists.\n";
qp += "Report exact command output. accept=true only if the remote suite is green.\n";
qp += "evidence = the test summary lines. gaps = failing test names or the block.\n";
let qa = agent(qp, #{
    label: "qa-terra",
    capability_mode: "all",
    output_schema: vote_schema,
});
let qa_ok = qa != () && qa.success && qa.output.accept == true;
if !qa_ok { all_ok = false; }
report += "qa=";
report += qa_ok.to_string();
report += "\n";
if qa != () && qa.success {
    report += json_encode(qa.output);
    report += "\n";
}
report += "all_ok=";
report += all_ok.to_string();
report += "\n";

let pth = write_scratch_file("barma-swarm.md", report);
complete(#{ path: pth, ok: all_ok, qa: qa_ok });
