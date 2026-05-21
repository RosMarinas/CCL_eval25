# Issue Tracker: GitHub

Issues and PRDs for this repo live in GitHub Issues at `github.com:RosMarinas/CCL_eval25`.

Use the `gh` CLI for issue operations from inside this repository.

## Conventions

- Create an issue: `gh issue create --title "..." --body "..."`
- Read an issue: `gh issue view <number> --comments`
- List issues: `gh issue list --state open --json number,title,body,labels,comments`
- Comment on an issue: `gh issue comment <number> --body "..."`
- Apply or remove labels: `gh issue edit <number> --add-label "..."` or `--remove-label "..."`
- Close an issue: `gh issue close <number> --comment "..."`

Infer the repository from `git remote -v`; `gh` does this automatically when run inside this clone.

## Skill Behavior

When a skill says "publish to the issue tracker", create a GitHub issue in `RosMarinas/CCL_eval25`.

When a skill says "fetch the relevant ticket", run `gh issue view <number> --comments`.
