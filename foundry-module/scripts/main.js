const MODULE_ID = "foundry-oracle";
const DEFAULT_API_URL = "https://foundry.crits4kids.net/oracle/ask";

function getChatCommandsApi() {
  const moduleIds = [
    "chat-commands",
    "chat-command-lib",
    "foundryvtt-chat-command-lib",
  ];
  for (const id of moduleIds) {
    const mod = game.modules.get(id);
    if (mod?.active && mod.api?.registerCommand) {
      return mod.api;
    }
  }
  return null;
}

async function askOracle(question) {
  const apiUrl = game.settings.get(MODULE_ID, "apiUrl");
  const response = await fetch(apiUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Oracle API error (${response.status}): ${text}`);
  }
  const data = await response.json();
  return data.answer || "No answer returned.";
}

async function postAnswer(content, messageData) {
  const speaker = ChatMessage.getSpeaker();
  const payload = {
    content,
    speaker,
    type: CONST.CHAT_MESSAGE_TYPES.OTHER,
  };
  if (messageData?.whisper) {
    payload.whisper = messageData.whisper;
  }
  await ChatMessage.create(payload);
}

function registerOracleCommand() {
  Hooks.on("chatCommandsReady", (commands) => {
    commands.register({
      name: "/oracle",
      module: "_chatcommands",
      aliases: ["/o"],
      description: game.i18n.localize(`${MODULE_ID}.command.oracle.description`),
      icon: "<i class='fas fa-hat-wizard'></i>",
      requiredRole: "NONE",
      callback: async (chat, parameters, messageData) => {
        const question = (parameters || "").trim();
        if (!question) {
          ui.notifications?.warn(
            game.i18n.localize(`${MODULE_ID}.errors.missingQuestion`)
          );
          return { content: "" };
        }
        try {
          const answer = await askOracle(question);
          await postAnswer(answer, messageData);
        } catch (error) {
          console.error(`${MODULE_ID} failed`, error);
          ui.notifications?.error(
            game.i18n.localize(`${MODULE_ID}.errors.requestFailed`)
          );
        }
        return { content: "" };
      },
      autocompleteCallback: () => [
        game.chatCommands.createInfoElement("Ask Foundry Oracle a question."),
      ],
      closeOnComplete: true,
    });
  });
}

Hooks.once("init", () => {
  game.settings.register(MODULE_ID, "apiUrl", {
    name: game.i18n.localize(`${MODULE_ID}.settings.apiUrl.name`),
    hint: game.i18n.localize(`${MODULE_ID}.settings.apiUrl.hint`),
    scope: "world",
    config: true,
    type: String,
    default: DEFAULT_API_URL,
  });
  registerOracleCommand();
});
