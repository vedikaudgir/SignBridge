# SignBridge HCI Design Rationale

This rationale maps the implemented interface to HCI principles. It distinguishes
requirements that define the product's safe, understandable baseline from
enhancements that are useful but not required for the primary recognition task.

## Mandatory design constraints

| Implemented decision | HCI principle | Rationale |
| --- | --- | --- |
| The primary view pairs the live camera preview with the backend-returned prediction and its confidence. | Match between system and real world; recognition rather than recall | The user can compare the current signing activity with the output without navigating away from the task. |
| Camera access begins only after a clearly labelled user action; Stop and Clear Stream remain explicit controls. | User control and freedom; informed consent | The user initiates and can end capture instead of the application starting a camera implicitly. |
| Camera permission, WebSocket connection, streaming, and errors are presented as distinct application states with a recovery path. | Visibility of system status; error recovery | A failed connection is not presented as successful streaming, and the user is told to check permission/backend availability and retry. |
| The interface states that frames are sent to the configured backend and that model confidence is not verified accuracy. | Informative feedback; honesty and trust | Users can understand the processing boundary and avoid interpreting confidence as measured correctness. |
| Low-confidence output receives a persistent, readable caution beside the prediction. | Error prevention; appropriate reliance | Uncertain output is not hidden behind reassuring visual treatment. |
| The accumulated text changes only after the user chooses Add to Sentence; Undo, Clear, and Reset provide reversal. | User control and freedom; error prevention | Users can review and edit the phrase instead of treating each prediction as final. |
| Privacy and Terms pages describe the inspected camera, backend, local preference, and optional translation flows, while qualifying deployment-dependent practices. | Informed consent; transparency | Documentation does not promise that infrastructure practices outside the repository are known. |
| Native buttons and links, visible keyboard focus, text status, responsive layout, and reduced-motion support are used. | Accessibility; universal design | Core actions remain operable and status information is not conveyed only through motion or color. |
| A consistent neutral palette, restrained rectangular controls, concise labels, and Lucide icons replace decorative gradients, emoji icons, and animated embellishments. | Consistency; reduced cognitive load | Visual emphasis is reserved for the live feed, prediction, and actionable status. |
| The page avoids unsupported model metrics, testimonials, user counts, and claims of accuracy. | Credibility; appropriate trust calibration | The interface reports only live backend output rather than decorative or fabricated evidence. |

## Optional enhancements

These are not required to use the core camera-to-prediction workflow and should
be added only when they solve a demonstrated user need:

| Enhancement | HCI principle | Current boundary |
| --- | --- | --- |
| Theme and text-size controls | Accessibility; personalization | Available as preferences; defaults do not block the primary task. |
| Translation and browser read-aloud | Flexibility and efficiency of use | Optional helpers. Translation sends accumulated text to the configured third-party MyMemory API and can be inaccurate. |
| Sentence building with Undo and Clear | User control; reversible actions | Text is accumulated only by explicit user action and is held in current browser session memory. |

## Known limitations

- Prediction quality depends on the configured backend/model and signing context;
  confidence is not verified accuracy.
- The current translation integration uses an external service and is not
  equivalent to sign-language interpretation.
- This documentation describes repository behavior. Backend hosting,
  network infrastructure, and service-provider retention practices may differ
  and need verification by the deployment operator.
