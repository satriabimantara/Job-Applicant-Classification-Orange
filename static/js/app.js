document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('applicant-form');
    const modelSelect = document.getElementById('model-select');
    const resultPanel = document.getElementById('result-panel');
    const btnPredict = document.getElementById('btn-predict');

    // Numeric feature keys
    const NUMERIC_KEYS = new Set(['YearsCode', 'YearsCodePro', 'PreviousSalary', 'ComputerSkills']);

    const safelyParseForm = (formData) => {
        const obj = {};
        formData.forEach((value, key) => {
            obj[key] = NUMERIC_KEYS.has(key) ? parseFloat(value) : value;
        });
        return obj;
    };

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Validation — model
        const selectedModel = modelSelect.value;
        if (!selectedModel) {
            showError("Please select a machine learning model first.");
            modelSelect.focus();
            return;
        }

        // Validation — required fields
        const requiredInputs = form.querySelectorAll('input[required], select[required]');
        let isValid = true;
        requiredInputs.forEach(input => {
            if (!input.value) {
                isValid = false;
                input.style.borderColor = '#ef4444';
                setTimeout(() => input.style.borderColor = '', 2000);
            }
        });
        if (!isValid) {
            showError("Please fill out all required fields.");
            return;
        }

        // Prepare payload
        const formData = new FormData(form);
        const features = safelyParseForm(formData);
        const payload = { model: selectedModel, features: features };

        // Loading state — clear immediately, no setTimeout race
        btnPredict.classList.add('loading');
        btnPredict.disabled = true;
        resultPanel.innerHTML = '';
        resultPanel.classList.remove('visible');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await response.json();
            console.log('[HireSight] Response:', data);

            if (!response.ok) {
                throw new Error(data.error || "Failed to get prediction");
            }

            // Guard against NaN / null values from backend
            if (data.prob_hired == null || data.prob_not_hired == null ||
                isNaN(data.prob_hired) || isNaN(data.prob_not_hired)) {
                throw new Error(
                    "Model returned invalid probability values. " +
                    "This may happen when certain input combinations are not supported by the selected model."
                );
            }

            renderResult(data, selectedModel);

        } catch (error) {
            console.error("[HireSight] Prediction Error:", error);
            showError(error.message);
        } finally {
            btnPredict.classList.remove('loading');
            btnPredict.disabled = false;
        }
    });

    function renderResult(data, modelName) {
        const isHired = data.prediction === 1;
        const probHired = data.prob_hired;
        const probNotHired = data.prob_not_hired;

        const badgeClass = isHired ? 'badge-hired' : 'badge-not-hired';
        const badgeIcon = isHired ? '🎉' : '⚠️';
        const badgeText = isHired ? 'EMPLOYED' : 'NOT EMPLOYED';

        const html = `
            <div class="mb-5 flex items-center justify-between">
                <h3 class="text-sm font-bold text-slate-800 flex items-center gap-2">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
                    AI Assessment Result
                </h3>
                <span class="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Model: ${modelName}</span>
            </div>

            <div class="text-center mb-6">
                <div class="${badgeClass} mb-3">
                    ${badgeIcon} ${badgeText}
                </div>
                <p class="text-sm text-slate-600 font-medium">
                    The model predicts this candidate is <b>${isHired ? 'likely' : 'unlikely'}</b> to be employed.
                </p>
            </div>

            <div class="space-y-4 bg-white/50 p-4 rounded-xl border border-slate-200/50">
                <div>
                    <div class="flex justify-between text-xs font-bold mb-2">
                        <span class="text-emerald-600">Employed (${probHired}%)</span>
                    </div>
                    <div class="prob-bar-track">
                        <div class="prob-bar-fill bar-hired" style="width: 0%"></div>
                    </div>
                </div>
                <div>
                    <div class="flex justify-between text-xs font-bold mb-2">
                        <span class="text-rose-500">Not Employed (${probNotHired}%)</span>
                    </div>
                    <div class="prob-bar-track">
                        <div class="prob-bar-fill bar-not-hired" style="width: 0%"></div>
                    </div>
                </div>
            </div>
        `;

        resultPanel.innerHTML = html;
        resultPanel.classList.add('visible');

        // Animate bars after DOM paint
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                const barHired = resultPanel.querySelector('.bar-hired');
                const barNot = resultPanel.querySelector('.bar-not-hired');
                if (barHired) barHired.style.width = probHired + '%';
                if (barNot) barNot.style.width = probNotHired + '%';
            });
        });

        // Scroll to result on mobile
        if (window.innerWidth < 1280) {
            resultPanel.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }

    function showError(message) {
        const html = `
            <div class="error-card">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="flex-shrink:0"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
                <div>
                    <h4 class="font-bold text-sm mb-1">Assessment Failed</h4>
                    <p class="text-xs opacity-90">${message}</p>
                </div>
            </div>
        `;
        resultPanel.innerHTML = html;
        resultPanel.classList.add('visible');
    }
});
