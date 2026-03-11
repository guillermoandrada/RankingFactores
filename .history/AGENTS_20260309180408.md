# AGENTS.md

## Project coding principles

This project must follow these principles in every task unless explicitly told otherwise:

- Preferred language: Python.
- Follow PEP 8 for naming, formatting, readability, and consistency.
- Apply SOLID principles.
- Follow Clean Code principles.
- Prioritize simplicity and avoid unnecessary complexity.
- Use DRY patterns and avoid duplicated logic.
- Prefer clear, explicit names over clever or abbreviated ones.
- Keep functions and methods small and focused on one responsibility.
- Prefer self-explanatory code over excessive comments.
- Control complexity by splitting logic into modules and layers.

## Symmetry rule

Symmetry is a first-class design constraint in this repository.

When implementing or changing something, always check whether there is an analogous module, class, method, schema, endpoint, config, test, or UI component that should follow the same pattern.

Prefer symmetrical design whenever reasonable:

- symmetrical folder structure
- symmetrical naming
- symmetrical method signatures
- symmetrical validation logic
- symmetrical error handling
- symmetrical return shapes
- symmetrical test structure
- symmetrical documentation
- symmetrical UI patterns
- symmetrical architecture across equivalent modules

If one module has a pattern that should logically exist in sibling modules, implement the corresponding pattern there too unless there is a strong reason not to.

Do not introduce one-off structures, one-off names, or one-off APIs without necessity.

If asymmetry is required, keep it minimal and explicitly explain why in the final summary.

## Architecture expectations

Before coding, identify the existing local pattern and extend it rather than inventing a new one.

Prefer:
- extension over ad hoc redesign
- abstraction over tight coupling
- composable helpers over repeated inline logic
- consistent interfaces over special cases
- layered organization over mixed concerns

Each class should have a single responsibility.
Depend on abstractions where possible, not concrete implementations.
Interfaces should be specific and focused.
New behavior should preferably be added by extension, not by destabilizing existing code.

## API design best practices

When working on APIs, prefer simple, resource-oriented designs with predictable behavior.

### General rules

- Keep routers thin.
- Keep business logic out of routers/controllers and place it in services/managers.
- Prefer a small set of clear endpoints per resource.
- As a default, aim for 4 or 5 endpoints per router, each aligned with a major CRUD action.
- Avoid adding special-purpose endpoints unless they are clearly justified.
- Use consistent request and response schemas across equivalent resources.
- Keep naming symmetrical across routers, DTOs, services, and tests.
- Prefer nouns for routes, not verbs.

### Default CRUD shape

Use this as the default pattern unless there is a strong reason not to:

- `GET /resources` → list resources
- `GET /resources/{id}` → retrieve one resource
- `POST /resources` → create
- `PUT /resources/{id}` or `PATCH /resources/{id}` → update
- `DELETE /resources/{id}` → delete

If both `PUT` and `PATCH` exist, their semantics must be clear and consistent.
Do not create multiple overlapping update endpoints without necessity.

### HTTP status code rules

Return proper HTTP status codes consistently.

Prefer:

- `200 OK` for successful reads and updates that return content
- `201 Created` for successful creations
- `204 No Content` for successful deletions or updates with no response body
- `400 Bad Request` for malformed requests
- `401 Unauthorized` for missing or invalid authentication
- `403 Forbidden` for authenticated users lacking permission
- `404 Not Found` when the resource does not exist
- `409 Conflict` for state conflicts, duplicates, or versioning conflicts
- `422 Unprocessable Entity` for validation failures when the payload is well-formed but invalid
- `500 Internal Server Error` only for unexpected server-side failures

Do not return `200` for everything.
Do not hide errors behind success responses.
Do not leak internal implementation details in error messages.

### Router/controller rules

Routers/controllers should:

- parse input
- validate basic request shape
- delegate work to a service/use-case layer
- translate domain/application errors into HTTP responses
- remain short and easy to scan

Routers/controllers should not:

- contain heavy business logic
- perform complex orchestration inline
- access persistence directly if a service/manager layer exists
- duplicate validation already handled elsewhere unless required by framework boundaries

### Request/response design

- Use stable, predictable response shapes.
- Keep response models consistent across similar endpoints.
- Return explicit error payloads with clear messages.
- Prefer pagination, filtering, and sorting conventions that are reusable across endpoints.
- Avoid mixing unrelated data into a single endpoint response unless there is a clear performance or UX reason.
- Prefer explicit fields over ambiguous nested structures.

### API symmetry rules

Equivalent resources should expose equivalent API patterns whenever reasonable.

Keep symmetrical:

- route naming
- CRUD coverage
- query parameter naming
- response envelopes
- validation behavior
- error response structure
- pagination format
- authentication and authorization patterns
- test coverage style
- documentation structure

If one resource has list/get/create/update/delete behavior, sibling resources should follow the same pattern unless there is a justified domain reason not to.

### API change discipline

- Prefer backward-compatible changes.
- Avoid breaking public contracts unless explicitly required.
- When changing an API, update schemas, validation, docs, and tests together.
- If an endpoint deviates from the repository default, explain why in the final summary.

## Implementation behavior

For every non-trivial task:

1. Inspect nearby files and identify the dominant repository pattern.
2. Reuse the existing architecture and naming conventions.
3. Check sibling modules for symmetry opportunities.
4. Implement the smallest clean solution that fits the architecture.
5. Update or add tests in the same style as existing tests.
6. Summarize what was changed and mention any intentional asymmetry.

## Code style defaults

- Use type hints where the codebase already uses them or where they improve clarity.
- Prefer explicit errors over silent fallbacks.
- Avoid overly generic utility layers unless they clearly reduce duplication.
- Avoid premature abstraction.
- Keep public APIs stable unless the task explicitly requires change.
- Avoid hidden side effects.
- Keep data models, schemas, and transformations consistent across similar entities.

## Repository conventions

When relevant, align work with this structure:

- `src/` for source code
- `tests/` for tests
- `docs/` for documentation

Respect the repository branching and delivery conventions already in use.

## Agent response expectations

When finishing a task, provide:

- what changed
- what pattern was followed
- what symmetric counterparts were checked
- any intentional deviation from symmetry and why