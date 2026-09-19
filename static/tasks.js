/* Day 13 task controls. Server state is authoritative; localStorage only selects a task. */
(() => {
    'use strict';
    const el = id => document.getElementById(id);
    const stages = ['planning', 'execution', 'validation', 'done'];
    const titles = {planning: 'Планирование', execution: 'Выполнение', validation: 'Проверка', done: 'Готово', paused: 'Пауза', failed: 'Ошибка'};
    const statuses = {pending: 'Ожидает', in_progress: 'Выполняется', completed: 'Выполнен', failed: 'Ошибка'};
    let tasks = [];
    let selected = null;
    const busy = new Set();
    let serverBusy = new Set();
    let profiles = [];
    let selectedProfile = '';
    let lastSessionProfile = null;
    let lastLogSignature = '';
    let polling = false;
    let deleting = false;
    let savedSelection = null;
    try { savedSelection = localStorage.getItem('day13-task'); } catch (_) { /* optional */ }

    async function api(url, body, method = body === undefined ? 'GET' : 'POST') {
        const response = await fetch(url, body === undefined ? {method} : {
            method, headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Не удалось выполнить действие');
        return data;
    }
    function remember(id) {
        selected = id;
        try { localStorage.setItem('day13-task', id); } catch (_) { /* optional */ }
    }
    function isBusy(id) { return busy.has(id) || serverBusy.has(id); }
    function renderLogs(task) {
        const logs = task.request_logs || [];
        const signature = task.task_id + JSON.stringify(logs);
        if (signature === lastLogSignature) return;
        lastLogSignature = signature;
        const open = new Set([...el('task-logs').querySelectorAll('details[open][data-request]')].map(item => item.dataset.request));
        el('task-logs').replaceChildren();
        el('task-log-count').textContent = '(' + logs.length + ')';
        const metered = logs.filter(log => log.metrics && log.metrics.tokens_source === 'api');
        const tokens = metered.reduce((sum, log) => sum + (Number(log.metrics.total_tokens) || 0), 0);
        const cost = metered.reduce((sum, log) => sum + (Number(log.metrics.cost_usd) || 0), 0);
        el('task-log-totals').textContent = logs.length ?
            'Токены по ответам API: ' + tokens + ' · оценка стоимости: $' + cost.toFixed(6) + ' · сохраняются последние 100 записей' :
            'После первого действия здесь появятся запрос, ответ, токены и применённый профиль.';
        const operationNames = {plan: 'Планирование', execute: 'Выполнение', validate: 'Проверка'};
        const logStatuses = {running: 'выполняется', success: 'успешно', error: 'ошибка', interrupted: 'прервано перезапуском'};
        logs.slice().reverse().forEach(log => {
            const card = document.createElement('details');
            card.className = 'task-request-log ' + log.status;
            card.dataset.request = log.request_id;
            card.open = open.has(log.request_id);
            const summary = document.createElement('summary');
            summary.textContent = new Date(log.started_at).toLocaleTimeString() + ' · ' + (operationNames[log.operation] || log.operation) +
                (log.step_id ? ' · шаг ' + log.step_id : '') + ' · ' + (logStatuses[log.status] || log.status) +
                ' · ' + (log.profile_id || 'без профиля') + ' · ' + Number(log.elapsed_seconds).toFixed(2) + ' с';
            card.append(summary);
            const info = document.createElement('p');
            const metrics = log.metrics || {};
            info.textContent = log.mode === 'llm' ?
                'Модель: ' + (log.model || 'ожидается') + ' · вход: ' + (metrics.prompt_tokens || 0) +
                ' · выход: ' + (metrics.completion_tokens || 0) +
                (metrics.tokens_source && metrics.tokens_source !== 'api' ? ' (оценка до ответа API)' : '') +
                ' · профиль в промпте: ' + (log.profile_used ? 'да' : 'нет') :
                (log.mode === 'local' ? 'Локальная проверка без запроса модели.' : 'Offline-сценарий: API и персонализация не используются.');
            card.append(info);
            if (log.error) {
                const error = document.createElement('p');
                error.className = 'task-log-error';
                error.textContent = log.error;
                card.append(error);
            }
            [
                ['Применённый профиль', {profile_id: log.profile_id, loaded_from_storage: log.profile_loaded,
                    included_in_prompt: log.profile_used, settings: log.profile_settings, overrides: log.profile_overrides}],
                ['Запрос к модели', log.request],
                ['Ответ модели', log.response],
            ].forEach(([label, value]) => {
                const detail = document.createElement('details');
                const heading = document.createElement('summary');
                heading.textContent = label;
                const pre = document.createElement('pre');
                pre.textContent = value == null ? 'Нет данных' : typeof value === 'string' ? value : JSON.stringify(value, null, 2);
                detail.append(heading, pre);
                card.append(detail);
            });
            el('task-logs').append(card);
        });
    }
    function renderTask() {
        const task = tasks.find(t => t.task_id === selected);
        el('task-empty').classList.toggle('hidden', !!task);
        el('task-detail').classList.toggle('hidden', !task);
        el('task-delete').disabled = deleting || !task || isBusy(selected);
        el('task-clear').disabled = deleting || !tasks.length || !!busy.size || !!serverBusy.size;
        if (!task) {
            el('task-progress').textContent = '';
            el('task-id').textContent = '';
            el('task-message').textContent = '';
            el('task-answer').value = '';
            el('task-logs').replaceChildren();
            lastLogSignature = '';
            return;
        }
        el('task-select').value = selected;
        el('task-id').textContent = task.task_id;
        el('task-goal-title').textContent = task.goal;
        const logs = task.request_logs || [];
        const applied = logs.slice().reverse().find(log => log.profile_used);
        const profile = applied && Object.keys(applied.profile_settings || {}).length ? applied.profile_settings :
            profiles.find(item => item.id === task.profile_id) || {};
        el('task-profile-title').textContent = 'Профиль задачи: ' + (task.profile_id || 'будет выбран при первом запросе');
        el('task-profile-summary').textContent = profile.id ?
            'Язык: ' + profile.language + ' · стиль: ' + profile.responseStyle +
            ' · формат: ' + profile.preferredFormat + ' · объём: ' + profile.responseLength :
            'Настройки появятся после первого запроса.';
        el('task-profile-note').textContent = applied ?
            'Показаны настройки последнего запроса. Профиль закреплён за задачей и сохраняется после перезапуска.' :
            logs.length && logs[logs.length - 1].mode === 'demo' ?
                'Offline-сценарий сохраняет выбранный профиль, но не применяет его к демонстрационному ответу.' :
                'Этот профиль будет использован при работе. Язык и формат из цели задачи имеют приоритет.';
        el('task-profile-json').textContent = JSON.stringify({
            profile_id: task.profile_id, settings: profile, overrides: applied ? applied.profile_overrides : {}
        }, null, 2);
        const completed = task.steps.filter(s => s.status === 'completed').length;
        el('task-progress').textContent = 'Выполнено: ' + completed + ' из ' + task.steps.length + '. Этап: ' + task.stage;
        el('task-flow').replaceChildren();
        (stages.includes(task.stage) ? stages : [...stages, task.stage]).forEach(stage => {
            const badge = document.createElement('span');
            badge.textContent = titles[stage] + ' · ' + stage;
            badge.classList.toggle('current', stage === task.stage);
            el('task-flow').append(badge);
        });
        const action = task.expected_action;
        el('task-next').textContent = action ? action.description : 'Задача завершена, результат сохранён.';
        const waiting = task.stage === 'planning' && action && action.actor === 'user';
        el('task-answer-form').classList.toggle('hidden', !waiting);
        el('task-question-label').textContent = waiting ? action.description : '';
        el('task-continue').disabled = isBusy(selected) || waiting || ['paused', 'done'].includes(task.stage);
        el('task-continue').textContent = isBusy(selected) ? 'Агент работает…' :
            task.stage === 'planning' ? 'Составить план' : task.stage === 'validation' ? 'Проверить результат' :
            task.stage === 'failed' ? 'Повторить шаг' : 'Выполнить следующий шаг';
        el('task-pause').disabled = ['paused', 'done'].includes(task.stage);
        el('task-resume').classList.toggle('hidden', task.stage !== 'paused');
        el('task-answer-form').querySelector('button').disabled = isBusy(selected);
        el('task-steps').replaceChildren();
        task.steps.forEach(step => {
            const card = document.createElement('div');
            card.className = 'task-step ' + step.status;
            const title = document.createElement('strong');
            title.textContent = step.id + '. ' + step.title + ' — ' + statuses[step.status] +
                (step.id === task.current_step_id ? ' · текущий шаг' : '');
            const description = document.createElement('p');
            description.textContent = step.description;
            card.append(title, description);
            if (step.result || step.error) {
                const result = document.createElement('pre');
                result.textContent = step.error || step.result;
                card.append(result);
            }
            el('task-steps').append(card);
        });
        el('task-final').classList.toggle('hidden', !task.final_result);
        el('task-final').textContent = task.final_result || '';
        el('task-inputs').textContent = task.questions.map(q => q.question + '\nОтвет: ' + (q.answer || 'ожидается')).join('\n\n') +
            '\n\nКритерии готовности:\n' + task.acceptance_criteria.join('\n');
        el('task-transitions').replaceChildren();
        task.transition_history.forEach(entry => {
            const row = document.createElement('li');
            row.textContent = entry.from_stage + ' → ' + entry.to_stage + ' · ' + entry.reason + ' · ' + new Date(entry.timestamp).toLocaleString();
            el('task-transitions').append(row);
        });
        el('task-json').textContent = JSON.stringify(task, null, 2);
        renderLogs(task);
        el('task-message').textContent = task.last_error || task.validation_issues.join('\n') ||
            (isBusy(selected) ? 'Запрос выполняется. Кнопка «Пауза» доступна.' : '');
    }
    async function refresh(preferred) {
        const data = await api('/api/tasks');
        tasks = data.tasks;
        serverBusy = new Set(data.busy_task_ids || []);
        profiles = data.available_profiles || [];
        const sessionProfile = data.current_profile && data.current_profile.profile.id;
        if (!selectedProfile || selectedProfile === lastSessionProfile) selectedProfile = sessionProfile;
        lastSessionProfile = sessionProfile;
        el('task-create-profile').replaceChildren();
        profiles.forEach(profile => {
            const option = document.createElement('option');
            option.value = profile.id;
            option.textContent = (profile.name || profile.id) + ' · ' + profile.language + ' · ' + profile.responseLength;
            el('task-create-profile').append(option);
        });
        el('task-create-profile').value = selectedProfile;
        el('task-mode').textContent = data.mode === 'demo' ? 'OFFLINE · учебный сценарий вакансии' : 'DeepSeek · сохранение в JSON';
        const candidate = preferred || selected || savedSelection;
        if (tasks.length) remember(tasks.some(t => t.task_id === candidate) ? candidate : tasks[0].task_id);
        else {
            selected = null;
            savedSelection = null;
            try { localStorage.removeItem('day13-task'); } catch (_) { /* optional */ }
        }
        el('task-select').replaceChildren();
        tasks.forEach(task => {
            const option = document.createElement('option');
            option.value = task.task_id;
            option.textContent = titles[task.stage] + ' · ' + task.goal.slice(0, 65);
            el('task-select').append(option);
        });
        renderTask();
    }
    async function act(action, body = {}) {
        const id = selected;
        if (!id) return;
        const isPause = action === 'pause';
        if (!isPause) busy.add(id);
        renderTask();
        let failure = null;
        try {
            await api('/api/tasks/' + encodeURIComponent(id) + '/' + action, body);
            if (body.answer) el('task-answer').value = '';
        } catch (error) {
            failure = error.message;
        } finally {
            if (!isPause) busy.delete(id);
            // Refresh on failure as well: the server may already have saved a result.
            await refresh().catch(error => { el('task-message').textContent = error.message; });
            if (failure) el('task-message').textContent = failure;
        }
    }
    window.refreshTasks = refresh;
    async function removeTasks(all) {
        const task = tasks.find(item => item.task_id === selected);
        if (deleting || (!all && !task) || (all && !tasks.length)) return;
        const message = all ?
            'Удалить все сохранённые задачи (' + tasks.length + ') вместе с результатами и логами? Это действие нельзя отменить.' :
            'Удалить задачу «' + task.goal + '» вместе с результатами и логами? Это действие нельзя отменить.';
        if (!window.confirm(message)) return;
        deleting = true;
        renderTask();
        let resultMessage;
        try {
            const url = all ? '/api/tasks' : '/api/tasks/' + encodeURIComponent(task.task_id);
            const result = await api(url, {confirm: true}, 'DELETE');
            el('task-answer').value = '';
            resultMessage = 'Удалено задач: ' + result.deleted_count + '.';
        } catch (error) {
            resultMessage = error.message;
        } finally {
            deleting = false;
            try { await refresh(); }
            catch (error) { resultMessage = error.message; }
            el('task-message').textContent = resultMessage;
        }
    }
    el('task-create-form').addEventListener('submit', async event => {
        event.preventDefault();
        const button = event.submitter;
        button.disabled = true;
        try {
            const data = await api('/api/tasks', {goal: el('task-goal').value, profile_id: el('task-create-profile').value});
            el('task-goal').value = '';
            await refresh(data.task.task_id);
        } catch (error) { el('task-message').textContent = error.message; }
        finally { button.disabled = false; }
    });
    el('task-select').addEventListener('change', () => { remember(el('task-select').value); renderTask(); });
    el('task-create-profile').addEventListener('change', () => { selectedProfile = el('task-create-profile').value; });
    el('task-refresh').addEventListener('click', () => refresh().catch(error => { el('task-message').textContent = error.message; }));
    el('task-delete').addEventListener('click', () => removeTasks(false));
    el('task-clear').addEventListener('click', () => removeTasks(true));
    el('task-continue').addEventListener('click', () => act('continue'));
    el('task-pause').addEventListener('click', () => act('pause'));
    el('task-resume').addEventListener('click', () => act('resume'));
    el('task-answer-form').addEventListener('submit', event => {
        event.preventDefault();
        act('continue', {answer: el('task-answer').value});
    });
    refresh().catch(error => { el('task-message').textContent = error.message; });
    setInterval(async () => {
        if (polling || (!busy.size && !serverBusy.size)) return;
        polling = true;
        try { await refresh(); }
        catch (error) { el('task-message').textContent = error.message; }
        finally { polling = false; }
    }, 1500);
})();
