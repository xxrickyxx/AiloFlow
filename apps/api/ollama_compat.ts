import { Express, Request, Response } from 'express';
import { DiscoveredModel } from '../../core/model/model_registry.js';
import { GenerationOptions } from '../../inference/base.js';
import { InferenceRegistry } from '../../inference/registry.js';

/**
 * Ollama-compatible surface.
 *
 * Most IDE integrations (Continue, Cline, Zed, the JetBrains AI plugins) speak
 * either the OpenAI protocol or Ollama's. AILOFlow already serves the former;
 * these routes add the latter so it can be selected anywhere Ollama is, by
 * pointing the tool at this port instead.
 *
 * The wire format is deliberately Ollama's, including its newline-delimited
 * JSON streaming and its nanosecond duration fields, because clients parse
 * those exact shapes.
 */

export interface OllamaCompatDeps {
  inference: InferenceRegistry;
  listModels: () => Promise<DiscoveredModel[]>;
  onGenerationStart: () => void;
  onGenerationEnd: () => void;
  version: string;
}

const MS_TO_NS = 1_000_000;

/**
 * End a request in a way the client can actually act on.
 *
 * Once streaming has started an Ollama client waits for an object carrying
 * `done: true`; closing the socket instead leaves it hanging until it gives up
 * with "did not receive done or success response in stream". So a mid-stream
 * failure is reported as a final NDJSON frame, and only a failure before the
 * first byte can still be a plain HTTP error.
 */
function finishWithError(res: Response, model: string | undefined, message: string): void {
  if (!res.headersSent) {
    res.status(400).json({ error: message });
    return;
  }

  res.write(
    `${JSON.stringify({
      model: model || '',
      created_at: new Date().toISOString(),
      message: { role: 'assistant', content: '' },
      done: true,
      done_reason: 'error',
      error: message,
    })}\n`
  );
  res.end();
}

function toOllamaTag(model: DiscoveredModel): string {
  // IDE settings fields expect something typeable, not an absolute path, so the
  // short alias is what we advertise. The full id still resolves on load.
  return model.alias;
}

function toOllamaModel(model: DiscoveredModel) {
  return {
    name: toOllamaTag(model),
    model: toOllamaTag(model),
    modified_at: model.modifiedAt || new Date(0).toISOString(),
    size: model.fileSizeBytes ?? 0,
    digest: '',
    details: {
      parent_model: '',
      format: 'gguf',
      family: model.source,
      families: [model.source],
      parameter_size: '',
      quantization_level: '',
    },
  };
}

export function registerOllamaCompatRoutes(app: Express, deps: OllamaCompatDeps): void {
  /**
   * Accept the Ollama endpoints under any base-URL prefix the user configured.
   *
   * IDE providers build requests as <base URL> + "/api/chat". Whatever the user
   * typed as the base URL ends up in front, so we see /api/api/chat when they
   * included /api, and /v1/api/chat when they pasted the OpenAI base URL.
   * Refusing those with a 404 that renders as an HTML error page inside the IDE
   * helps nobody, so the path is normalised to the last /api/ segment. Routes
   * that do not contain /api/ at all — the whole /v1 surface — are untouched.
   */
  app.use((req, _res, next) => {
    const queryStart = req.url.indexOf('?');
    const pathname = queryStart === -1 ? req.url : req.url.slice(0, queryStart);
    const query = queryStart === -1 ? '' : req.url.slice(queryStart);

    // Normalise on the *last* /api/ segment: that collapses both /v1/api/chat
    // and /api/api/chat onto /api/chat, while a correct /api/chat (marker at 0)
    // is left alone.
    const marker = pathname.lastIndexOf('/api/');
    if (marker > 0) req.url = pathname.slice(marker) + query;

    next();
  });

  // Real Ollama answers GET / with this exact plain-text banner, and several
  // clients probe it to decide whether the server is alive.
  app.get('/', (_req, res) => {
    res.type('text/plain').send('Ollama is running');
  });

  // A bare GET /api is someone exploring with a browser: point them somewhere
  // useful instead of "Cannot GET /api".
  app.get('/api', (_req, res) => {
    res.json({
      server: 'ailoflow',
      message: 'Ollama-compatible API. Use this host WITHOUT the /api suffix as the base URL in your IDE.',
      endpoints: ['/api/version', '/api/tags', '/api/chat', '/api/generate', '/api/show', '/api/ps'],
      openai_compatible: '/v1',
    });
  });

  app.get('/api/version', (_req, res) => {
    // Reported as an Ollama-compatible version so clients that gate on it work,
    // alongside our own identity.
    res.json({ version: '0.6.0', server: 'ailoflow', ailoflow_version: deps.version });
  });

  app.get('/api/tags', async (_req, res) => {
    try {
      const models = await deps.listModels();
      res.json({ models: models.map(toOllamaModel) });
    } catch (err) {
      res.status(500).json({ error: (err as Error).message });
    }
  });

  app.get('/api/ps', (_req, res) => {
    const state = deps.inference.getState();
    res.json({
      models: state
        ? [
            {
              ...toOllamaModel(state.model),
              expires_at: '0001-01-01T00:00:00Z',
              size_vram: 0,
            },
          ]
        : [],
    });
  });

  app.post('/api/show', async (req, res) => {
    const name = req.body?.model || req.body?.name;
    if (!name) return res.status(400).json({ error: 'field "model" is required' });

    try {
      const models = await deps.listModels();
      const model = models.find((m) => m.id === name || toOllamaTag(m) === name);
      if (!model) return res.status(404).json({ error: `model "${name}" not found` });

      res.json({
        license: '',
        modelfile: '',
        parameters: '',
        template: '',
        details: toOllamaModel(model).details,
        model_info: {
          'general.architecture': model.source,
          'general.file_type': 0,
          'ailoflow.file_path': model.filePath,
          'ailoflow.runnable_with': model.runnableWith,
        },
      });
    } catch (err) {
      res.status(500).json({ error: (err as Error).message });
    }
  });

  const runGeneration = async (
    res: Response,
    modelName: string,
    options: GenerationOptions,
    stream: boolean,
    shape: 'chat' | 'generate'
  ): Promise<void> => {
    // Accept either the full id or the alias, and only reload when the request
    // actually names a different model than the one already resident.
    const current = deps.inference.getState()?.model;
    if (!current || (current.id !== modelName && current.alias !== modelName)) {
      await deps.inference.loadModel(modelName);
    }

    const backend = deps.inference.getActiveBackend();
    const state = deps.inference.getState();
    if (!backend || !state) throw new Error('no model loaded');

    // An IDE that cancels a request, or a user closing the panel, closes this
    // response. Without forwarding that the engine keeps generating — a
    // reasoning model can then hold the GPU for many minutes producing tokens
    // nobody will ever read.
    const abort = new AbortController();
    res.on('close', () => abort.abort());
    options.signal = abort.signal;

    deps.onGenerationStart();
    const createdAt = () => new Date().toISOString();

    try {
      if (stream) {
        // Ollama streams newline-delimited JSON objects, not SSE frames.
        res.setHeader('Content-Type', 'application/x-ndjson');
        res.flushHeaders?.();

        const result = await backend.generateStream(options, (token) => {
          if (token.isFinished || !token.token) return;

          // A reasoning model's chain of thought belongs in `thinking`, the
          // field Ollama uses for it. Emitting it as `content` would splice
          // paragraphs of deliberation into the assistant's actual answer.
          const isReasoning = token.kind === 'reasoning';
          const chunk =
            shape === 'chat'
              ? {
                  model: modelName,
                  created_at: createdAt(),
                  message: isReasoning
                    ? { role: 'assistant', content: '', thinking: token.token }
                    : { role: 'assistant', content: token.token },
                  done: false,
                }
              : {
                  model: modelName,
                  created_at: createdAt(),
                  ...(isReasoning ? { thinking: token.token, response: '' } : { response: token.token }),
                  done: false,
                };
          res.write(`${JSON.stringify(chunk)}\n`);
        });

        deps.inference.recordMetrics(result.metrics);

        const m = result.metrics;
        const final = {
          model: modelName,
          created_at: createdAt(),
          ...(shape === 'chat' ? { message: { role: 'assistant', content: '' } } : { response: '' }),
          done: true,
          done_reason: 'stop',
          total_duration: Math.round(m.totalDurationMs * MS_TO_NS),
          load_duration: 0,
          prompt_eval_count: m.promptTokens ?? 0,
          prompt_eval_duration:
            m.promptTokens && m.promptTokensPerSecond
              ? Math.round((m.promptTokens / m.promptTokensPerSecond) * 1000 * MS_TO_NS)
              : 0,
          eval_count: m.completionTokens ?? 0,
          eval_duration:
            m.completionTokens && m.tokensPerSecond
              ? Math.round((m.completionTokens / m.tokensPerSecond) * 1000 * MS_TO_NS)
              : 0,
        };
        res.write(`${JSON.stringify(final)}\n`);
        res.end();
      } else {
        const result = await backend.generateStream(options, () => { /* buffered */ });
        deps.inference.recordMetrics(result.metrics);
        const m = result.metrics;

        res.json({
          model: modelName,
          created_at: createdAt(),
          ...(shape === 'chat'
            ? {
                message: {
                  role: 'assistant',
                  content: result.text,
                  ...(result.reasoning ? { thinking: result.reasoning } : {}),
                },
              }
            : { response: result.text, ...(result.reasoning ? { thinking: result.reasoning } : {}) }),
          done: true,
          done_reason: 'stop',
          total_duration: Math.round(m.totalDurationMs * MS_TO_NS),
          prompt_eval_count: m.promptTokens ?? 0,
          eval_count: m.completionTokens ?? 0,
          eval_duration:
            m.completionTokens && m.tokensPerSecond
              ? Math.round((m.completionTokens / m.tokensPerSecond) * 1000 * MS_TO_NS)
              : 0,
        });
      }
    } finally {
      deps.onGenerationEnd();
    }
  };

  const mapOptions = (body: Record<string, any>): Partial<GenerationOptions> => {
    const o = body.options || {};
    return {
      temperature: o.temperature,
      topP: o.top_p,
      topK: o.top_k,
      repetitionPenalty: o.repeat_penalty,
      maxTokens: o.num_predict,
      contextLength: o.num_ctx,
      seed: o.seed,
      stopSequences: Array.isArray(o.stop) ? o.stop : undefined,
    };
  };

  app.post('/api/chat', async (req: Request, res: Response) => {
    const body = req.body || {};
    const messages: Array<{ role: string; content: string }> = body.messages || [];
    const last = [...messages].reverse().find((m) => m.role === 'user');

    if (!body.model) return res.status(400).json({ error: 'field "model" is required' });
    if (!last) return res.status(400).json({ error: 'at least one user message is required' });

    try {
      await runGeneration(
        res,
        body.model,
        {
          prompt: last.content,
          systemPrompt: messages.find((m) => m.role === 'system')?.content,
          history: messages
            .filter((m) => m.role === 'user' || m.role === 'assistant')
            .slice(0, -1) as Array<{ role: 'user' | 'assistant'; content: string }>,
          ...mapOptions(body),
        },
        body.stream !== false,
        'chat'
      );
    } catch (err) {
      // A client that disconnected is not an error worth reporting back to it.
      if ((err as Error).name === 'AbortError' || res.destroyed) return;
      finishWithError(res, body.model, (err as Error).message);
    }
  });

  app.post('/api/generate', async (req: Request, res: Response) => {
    const body = req.body || {};
    if (!body.model) return res.status(400).json({ error: 'field "model" is required' });
    if (typeof body.prompt !== 'string') return res.status(400).json({ error: 'field "prompt" is required' });

    try {
      await runGeneration(
        res,
        body.model,
        { prompt: body.prompt, systemPrompt: body.system, ...mapOptions(body) },
        body.stream !== false,
        'generate'
      );
    } catch (err) {
      finishWithError(res, body.model, (err as Error).message);
    }
  });

  // Operations that only make sense against Ollama's own registry. Answering
  // with a clear 501 beats a silent failure inside an IDE.
  for (const route of ['/api/pull', '/api/push', '/api/create', '/api/copy', '/api/delete']) {
    app.post(route, (_req, res) => {
      res.status(501).json({
        error:
          `${route} is not supported: AILOFlow manages models through its own catalog. ` +
          'Use the Catalogo tab or POST /v1/downloads to fetch a model.',
      });
    });
  }

  app.post('/api/embeddings', (_req, res) => {
    res.status(501).json({ error: 'Embeddings are not implemented yet.' });
  });
}
