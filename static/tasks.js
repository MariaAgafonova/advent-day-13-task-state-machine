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
    let savedSelection = null;
    try { savedSelection = localStorage.getItem('day13-task'); } catch (_) { /* optional */ }

    async function api(url, body) {
        const response = await fetch(url, body === undefined ? {} : {
            method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Не удалось выполнить действие');
        return data;
    }
    function remember(id) {
        selected = id;
        try { localStorage.setItem('day13-task', id); } catch (_) { /* optional */ }
    }
    function renderTask() {
        const task = tasks.find(t => t.task_id === selected);
        el('task-empty').classList.toggle('hidden', !!task);
        el('task-detail').classList.toggle('hidden', !task);
        if (!task) return;
        el('task-select').value = selected;
        el('task-id').textContent = task.task_id;
        el('task-goal-title').textContent = task.goal;
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
        el('task-continue').disabled = busy.has(selected) || waiting || ['paused', 'done'].includes(task.stage);
        el('task-continue').textContent = busy.has(selected) ? 'Агент работает…' :
            task.stage === 'planning' ? 'Составить план' : task.stage === 'validation' ? 'Проверить результат' :
            task.stage === 'failed' ? 'Повторить шаг' : 'Выполнить следующий шаг';
        el('task-pause').disabled = ['paused', 'done'].includes(task.stage);
        el('task-resume').classList.toggle('hidden', task.stage !== 'paused');
        el('task-answer-form').querySelector('button').disabled = busy.has(selected);
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
        el('task-message').textContent = task.last_error || task.validation_issues.join('\n') ||
            (busy.has(selected) ? 'Запрос выполняется. Кнопка «Пауза» доступна.' : '');
    }
    async function refresh(preferred) {
        const data = await api('/api/tasks');
        tasks = data.tasks;
        el('task-mode').textContent = data.mode === 'demo' ? 'OFFLINE · учебный сценарий вакансии' : 'DeepSeek · сохранение в JSON';
        const candidate = preferred || selected || savedSelection;
        if (tasks.length) remember(tasks.some(t => t.task_id === candidate) ? candidate : tasks[0].task_id);
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
    el('task-create-form').addEventListener('submit', async event => {
        event.preventDefault();
        const button = event.submitter;
        button.disabled = true;
        try {
            const data = await api('/api/tasks', {goal: el('task-goal').value});
            el('task-goal').value = '';
            await refresh(data.task.task_id);
        } catch (error) { el('task-message').textContent = error.message; }
        finally { button.disabled = false; }
    });
    el('task-select').addEventListener('change', () => { remember(el('task-select').value); renderTask(); });
    el('task-refresh').addEventListener('click', () => refresh().catch(error => { el('task-message').textContent = error.message; }));
    el('task-continue').addEventListener('click', () => act('continue'));
    el('task-pause').addEventListener('click', () => act('pause'));
    el('task-resume').addEventListener('click', () => act('resume'));
    el('task-answer-form').addEventListener('submit', event => {
        event.preventDefault();
        act('continue', {answer: el('task-answer').value});
    });
    refresh().catch(error => { el('task-message').textContent = error.message; });
})();
