const $ = (s, root=document) => root.querySelector(s)
const $$ = (s, root=document) => [...root.querySelectorAll(s)]
const pct = (v,d=1) => `${(Number(v||0)*100).toFixed(d)}%`
const num = (v,d=0) => Number(v||0).toLocaleString(undefined,{maximumFractionDigits:d,minimumFractionDigits:d})
const compact = v => new Intl.NumberFormat('en-IN',{notation:'compact',maximumFractionDigits:1}).format(Number(v||0))
const money = v => `₹${compact(v)}`
const esc = v => String(v??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]))
const pretty = s => String(s||'').replace(/^num__|^cat__/,'').replaceAll('_',' ').replace(/\b\w/g,c=>c.toUpperCase())

let analysisData=null, baselineAnalysis=null, optimizedPolicy=null, queueData=[], enhancedReady=false, enhancedInfo=null, legacyInfo=null, currentMode='legacy'
const pageMeta={
 command:['Loan approval analytics','Project Dashboard'],policy:['Policy and staffing analysis','Policy Analysis'],queue:['Manual review workflow','Review Queue'],segments:['Segment and bottleneck analysis','Segment Analysis'],score:['Loan approval prediction','Loan Prediction'],model:['Model evaluation','Model Performance'],story:['Internship project overview','Project Overview']
}

function go(page){
 $$('.page').forEach(x=>x.classList.remove('active')); $(`#page-${page}`)?.classList.add('active')
 $$('.side-nav button').forEach(x=>x.classList.toggle('active',x.dataset.page===page))
 $('#pageEyebrow').textContent=pageMeta[page]?.[0]||''; $('#pageName').textContent=pageMeta[page]?.[1]||''
 if(page==='model') renderModelEvidence(); if(page==='queue') renderQueue(); if(page==='segments') renderSegments()
 window.scrollTo({top:0,behavior:'smooth'})
}
$$('[data-page]').forEach(b=>b.addEventListener('click',()=>go(b.dataset.page)))
$$('[data-go]').forEach(b=>b.addEventListener('click',()=>go(b.dataset.go)))

function scenarioPayload(){
 return {
  approve_threshold:Number($('#approveThreshold').value)/100,
  reject_threshold:Number($('#rejectThreshold').value)/100,
  analysts:Number($('#analystsInput').value),
  productive_hours:Number($('#productiveHoursInput').value),
  volume_multiplier:Number($('#volumeMultiplier').value)/100,
  segment_dimension:$('#segmentDimension').value
 }
}
function updateControlLabels(){
 $('#approveThresholdLabel').textContent=`${$('#approveThreshold').value}%`
 $('#rejectThresholdLabel').textContent=`${$('#rejectThreshold').value}%`
 $('#volumeLabel').textContent=`${(Number($('#volumeMultiplier').value)/100).toFixed(1)}×`
 $('#agreementLabel').textContent=`${$('#agreementGuardrail').value}%`
}
['approveThreshold','rejectThreshold','volumeMultiplier','agreementGuardrail'].forEach(id=>$('#'+id).addEventListener('input',updateControlLabels))

async function postJSON(url,payload){
 const res=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)})
 const body=await res.json(); if(!res.ok) throw new Error(typeof body.detail==='string'?body.detail:JSON.stringify(body.detail||body)); return body
}

async function runScenario({preserveBaseline=true}={}){
 const btn=$('#runScenarioBtn'); btn.disabled=true; btn.textContent='Running…'
 try{
  const data=await postJSON('/api/analysis/run_business_analysis',scenarioPayload())
  analysisData=data; queueData=data.queue||[]; if(!baselineAnalysis||!preserveBaseline) baselineAnalysis=data
  renderCommand(); renderPolicy(); renderQueue(); renderSegments()
 }catch(err){
  console.error(err); alert(`Scenario failed: ${err.message}`)
 }finally{btn.disabled=false;btn.textContent='Run scenario'}
}
$('#runScenarioBtn').addEventListener('click',()=>runScenario())
$('#refreshBtn').addEventListener('click',()=>runScenario())
$('#segmentDimension').addEventListener('change',()=>runScenario())

function kpi(label,value,note,cls=''){return `<div class="kpi ${cls}"><span>${esc(label)}</span><strong>${esc(value)}</strong><small>${esc(note)}</small></div>`}
function renderCommand(){
 if(!analysisData)return
 const p=analysisData.policy, summary=analysisData.dataset_summary
 $('#commandKpis').classList.remove('skeleton-grid')
 $('#commandKpis').innerHTML=[
  kpi('Applications',num(summary.applications),'42-day simulated operating window'),
  kpi('Auto-decision rate',pct(p.auto_decision_rate),'applications routed without human review','accent'),
  kpi('Auto agreement',pct(p.auto_historical_agreement),'vs historical decisions on automated cases','good'),
  kpi('Manual review',pct(p.manual_review_rate),`${num(p.manual_review)} cases in selected policy`),
  kpi('Capacity utilization',pct(p.capacity_utilization),p.capacity_status,p.capacity_status==='Over capacity'?'warn':''),
  kpi('Analyst hours saved',num(p.analyst_hours_saved,1),'vs reviewing every application manually')
 ].join('')
 $('#executiveBrief').classList.remove('loading-lines'); $('#executiveBrief').innerHTML=(analysisData.analysis_summary||[]).map((x,i)=>`<div class="brief-item"><b>${i+1}</b><span>${esc(x)}</span></div>`).join('')
 const max=Math.max(...analysisData.funnel.map(x=>x.count),1)
 $('#funnel').classList.remove('loading-lines'); $('#funnel').innerHTML=analysisData.funnel.map(x=>`<div class="funnel-row"><div><header><span>${esc(x.stage)}</span><span>${num(x.count)}</span></header><div class="track"><i style="width:${Math.max(2,x.count/max*100)}%"></i></div></div><small>${pct(x.rate)}</small></div>`).join('')
 renderTrend(analysisData.trends||[])
 $('#opportunityList').classList.remove('loading-lines'); $('#opportunityList').innerHTML=(analysisData.segments||[]).slice(0,4).map((s,i)=>`<div class="opportunity"><span class="rank">${i+1}</span><div><h4>${esc(s.segment)}</h4><p>${pct(s.manual_review_rate)} manual · ${num(s.avg_processing_hours,1)}h TAT · ${pct(s.sla_breach_rate)} SLA breach</p></div><strong>${num(s.applications)} apps</strong></div>`).join('')
}

function renderTrend(rows){
 const el=$('#trendChart'); if(!rows.length){el.innerHTML='<p>No trend data.</p>';return}
 const W=800,H=220,pad=24; const maxApps=Math.max(...rows.map(r=>r.applications),1)
 const x=i=>pad+i*(W-pad*2)/Math.max(1,rows.length-1); const yApps=v=>H-pad-(v/maxApps)*(H-pad*2)
 const yRate=v=>H-pad-Number(v)*(H-pad*2)
 const bars=rows.map((r,i)=>{const bw=Math.max(3,(W-pad*2)/rows.length*.55);return `<rect x="${x(i)-bw/2}" y="${yApps(r.applications)}" width="${bw}" height="${H-pad-yApps(r.applications)}" rx="2" fill="#7c5cff" opacity=".32"/>`}).join('')
 const points=rows.map((r,i)=>`${x(i)},${yRate(r.manual_review_rate)}`).join(' ')
 const labels=[0,Math.floor(rows.length/2),rows.length-1].map(i=>`<text x="${x(i)}" y="${H-5}" text-anchor="middle" font-size="9" fill="#8a919c">${esc(rows[i].date.slice(5))}</text>`).join('')
 el.innerHTML=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Applications and manual review trend"><line x1="${pad}" y1="${H-pad}" x2="${W-pad}" y2="${H-pad}" stroke="#e3e7ec"/>${bars}<polyline fill="none" stroke="#35d6a6" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" points="${points}"/>${labels}</svg>`
}

function renderPolicy(){
 if(!analysisData)return
 const p=analysisData.policy
 const cls=p.capacity_status==='Over capacity'?'bad':p.capacity_status==='Tight'?'warn':''
 $('#capacityBadge').className=`status-badge ${cls}`; $('#capacityBadge').textContent=p.capacity_status
 $('#scenarioKpis').innerHTML=[
  ['Automation',pct(p.auto_decision_rate)],['Auto agreement',pct(p.auto_historical_agreement)],['Manual review',pct(p.manual_review_rate)],['Max sustainable demand',`${num(p.max_sustainable_volume_multiplier,2)}×`]
 ].map(([a,b])=>`<div class="scenario-stat"><span>${a}</span><strong>${b}</strong></div>`).join('')
 const total=p.applications||1, a=p.auto_approve/total*100,m=p.manual_review/total*100,r=p.auto_reject/total*100
 $('#decisionMix').innerHTML=`<div class="approve" style="width:${a}%" title="Auto approve">${a>8?`${a.toFixed(0)}%`:''}</div><div class="manual" style="width:${m}%" title="Manual review">${m>8?`${m.toFixed(0)}%`:''}</div><div class="reject" style="width:${r}%" title="Auto reject">${r>8?`${r.toFixed(0)}%`:''}</div>`
 const util=p.capacity_utilization; $('#capacityLabel').textContent=pct(util); $('#capacityBar').style.width=`${Math.min(100,util*100)}%`; $('#capacityBar').style.background=util>1?'#ef6b68':util>.85?'#f0ad4e':'#35d6a6'
 $('#capacityNote').textContent=p.backlog_growth_cases_per_day>0?`Backlog grows by ~${num(p.backlog_growth_cases_per_day,1)} cases/day under this scenario.`:`Capacity buffer remains; estimated throughput is ${num(p.daily_case_capacity,1)} manual cases/day.`
 renderPolicyComparison(); renderStress()
}
function rowPolicy(name,p){if(!p)return'';return `<tr><td><strong>${esc(name)}</strong></td><td>${pct(p.auto_decision_rate)}</td><td>${pct(p.auto_historical_agreement)}</td><td>${pct(p.manual_review_rate)}</td><td>${num(p.analyst_hours_saved,1)}h</td></tr>`}
function renderPolicyComparison(){
 if(!analysisData)return
 const opt=optimizedPolicy?.policy || analysisData.optimized_policy_90pct_guardrail?.policy
 $('#policyCompareRows').innerHTML=rowPolicy('Baseline 75/25',baselineAnalysis?.policy)+rowPolicy('Current',analysisData.policy)+rowPolicy('Optimized',opt)
}
function renderStress(){
 if(!analysisData)return
 const p=analysisData.policy, baseUtil=p.capacity_utilization/Math.max(.01,p.volume_multiplier)
 const multipliers=[.75,1,1.5,2]
 $('#stressGrid').innerHTML=multipliers.map(m=>{const util=baseUtil*m, cls=util>1?'bad':'good', status=util>1?'Backlog risk':util>.85?'Tight':'Healthy';return `<div class="stress-card ${cls}"><span>${m.toFixed(2)}× demand</span><strong>${pct(util)}</strong><small>${status} · capacity utilization</small></div>`}).join('')
}

$('#optimizePolicyBtn').addEventListener('click',async()=>{
 const b=$('#optimizePolicyBtn'); b.disabled=true;b.textContent='Optimizing…'
 try{
  optimizedPolicy=await postJSON('/api/analysis/optimize-policy',{minimum_auto_agreement:Number($('#agreementGuardrail').value)/100,analysts:Number($('#analystsInput').value),productive_hours:Number($('#productiveHoursInput').value),volume_multiplier:Number($('#volumeMultiplier').value)/100})
  const p=optimizedPolicy.policy; const call=$('#optimizerCallout');call.classList.remove('hidden')
  call.innerHTML=`<div><span class="eyebrow">Recommended operating policy</span><h3>${pct(p.reject_threshold,0)} reject / ${pct(p.approve_threshold,0)} approve</h3><p>Maximizes automated coverage while meeting the selected historical-agreement guardrail. Expected automation ${pct(p.auto_decision_rate)}, agreement ${pct(p.auto_historical_agreement)}, manual review ${pct(p.manual_review_rate)}.</p></div><button class="primary" id="applyOptimized">Apply policy</button>`
  $('#applyOptimized').addEventListener('click',()=>{$('#approveThreshold').value=Math.round(p.approve_threshold*100);$('#rejectThreshold').value=Math.round(p.reject_threshold*100);updateControlLabels();runScenario()})
  renderPolicyComparison()
 }catch(err){alert(`Optimizer failed: ${err.message}`)}finally{b.disabled=false;b.textContent='Find best policy'}
})

function renderQueue(){
 if(!analysisData)return
 let rows=queueData.slice(); const q=$('#queueSearch').value.trim().toLowerCase(), f=$('#queuePriorityFilter').value
 if(q)rows=rows.filter(x=>[x.application_id,x.channel,x.priority_reason,x.property_area,x.analyst].some(v=>String(v||'').toLowerCase().includes(q)))
 if(f==='high')rows=rows.filter(x=>x.priority>=70); if(f==='sla')rows=rows.filter(x=>x.sla_breached); if(f==='docs')rows=rows.filter(x=>x.priority_reason==='Document friction')
 $('#queueCount').textContent=num(rows.length)
 $('#queueRows').innerHTML=rows.length?rows.map(r=>`<tr><td><span class="priority-chip ${r.priority>=70?'high':r.priority>=50?'med':'low'}">${num(r.priority,0)}</span></td><td><strong>${esc(r.application_id)}</strong><br><span class="tiny-pill">${esc(r.analyst)}</span></td><td>${esc(r.priority_reason)} ${r.sla_breached?'<span class="tiny-pill danger-pill">SLA</span>':''}</td><td>${pct(r.approval_probability)}</td><td>${money(r.loan_value)}</td><td>${esc(r.channel)}</td><td>${pct(r.document_completeness,0)}</td><td>${num(r.processing_hours,1)}h</td><td><button class="row-action" data-case="${esc(r.application_id)}">Inspect →</button></td></tr>`).join(''):'<tr><td colspan="9">No cases match this filter.</td></tr>'
 $$('[data-case]').forEach(b=>b.addEventListener('click',()=>openCase(b.dataset.case)))
}
$('#queueSearch').addEventListener('input',renderQueue); $('#queuePriorityFilter').addEventListener('change',renderQueue)

async function openCase(id){
 const drawer=$('#caseDrawer');drawer.classList.remove('hidden');$('#caseDrawerBody').innerHTML='<p>Loading case…</p>'
 try{
  const res=await fetch(`/api/analysis/case/${encodeURIComponent(id)}`); const c=await res.json(); if(!res.ok)throw new Error(c.detail||'Case not found')
  $('#caseDrawerBody').innerHTML=`<span class="eyebrow">Application details</span><h2>${esc(c.application_id)}</h2><p>${esc(c.priority_reason)} · ${esc(c.channel)} · ${esc(c.analyst)}</p><div class="drawer-grid"><div class="drawer-stat"><span>Approval probability</span><strong>${pct(c.approval_probability)}</strong></div><div class="drawer-stat"><span>Historical outcome</span><strong>${esc(c.historical_outcome)}</strong></div><div class="drawer-stat"><span>Priority score</span><strong>${num(c.priority_score,0)}/100</strong></div><div class="drawer-stat"><span>Loan value</span><strong>${money(c.loan_value)}</strong></div><div class="drawer-stat"><span>Processing time</span><strong>${num(c.processing_hours,1)}h</strong></div><div class="drawer-stat"><span>Documents</span><strong>${pct(c.document_completeness,0)}</strong></div></div><div class="result-section"><h4>Applicant snapshot</h4><div class="drawer-grid">${Object.entries(c.applicant).map(([k,v])=>`<div class="drawer-stat"><span>${pretty(k)}</span><strong>${esc(v??'Missing')}</strong></div>`).join('')}</div></div><div class="audit-box"><span class="eyebrow">Prediction details</span><p><strong>${esc(c.audit.model)}</strong> · ${esc(c.audit.model_version)}</p><small>Target: ${esc(c.audit.source_target)}. Synthetic operations fields: ${esc(c.audit.simulation_fields.join(', '))}.</small></div><p class="microcopy" style="margin-top:12px">${esc(c.disclaimer)}</p>`
 }catch(err){$('#caseDrawerBody').innerHTML=`<p class="error">${esc(err.message)}</p>`}
}
$('#closeDrawer').addEventListener('click',()=>$('#caseDrawer').classList.add('hidden')); $('#caseDrawer').addEventListener('click',e=>{if(e.target===$('#caseDrawer'))$('#caseDrawer').classList.add('hidden')})

function renderSegments(){
 if(!analysisData)return
 const rows=analysisData.segments||[], top=rows.slice(0,3)
 $('#rootCauseCards').innerHTML=top.map((s,i)=>`<div class="root-card"><span>#${i+1} operational opportunity</span><strong>${esc(s.segment)}</strong><p>${num(s.applications)} apps · ${pct(s.manual_review_rate)} manual · ${num(s.avg_processing_hours,1)}h TAT · ${pct(s.document_completeness,0)} document completeness.</p></div>`).join('')
 $('#segmentRows').innerHTML=rows.map(s=>`<tr><td><strong>${esc(s.segment)}</strong></td><td>${num(s.applications)}</td><td>${pct(s.manual_review_rate)}</td><td>${num(s.avg_processing_hours,1)}h</td><td>${pct(s.sla_breach_rate)}</td><td>${pct(s.document_completeness,0)}</td><td>${pct(s.historical_approval_rate)}</td><td>${pct(s.auto_historical_agreement)}</td></tr>`).join('')
 const q=analysisData.data_quality||{}, miss=q.missing_by_field||[], max=Math.max(...miss.map(x=>x.missing),1)
 $('#qualityPanel').innerHTML=`<div class="quality-stat-grid"><div class="quality-stat"><strong>${num(q.rows)}</strong><span>source rows</span></div><div class="quality-stat"><strong>${num(q.missing_cells)}</strong><span>missing cells</span></div><div class="quality-stat"><strong>${num(q.duplicate_rows)}</strong><span>duplicate rows</span></div></div><div class="missing-list">${miss.slice(0,6).map(x=>`<div class="missing-row"><span>${esc(x.field)}</span><div class="mini-track"><i style="width:${x.missing/max*100}%"></i></div><strong>${num(x.missing)}</strong></div>`).join('')}</div>`
 const first=rows[0], second=rows[1]
 const narrative=[]
 if(first)narrative.push(`${first.segment} has the highest composite operations opportunity score: ${pct(first.manual_review_rate)} manual review with ${num(first.avg_processing_hours,1)}h average processing time.`)
 if(second)narrative.push(`${second.segment} is the next segment to inspect; compare its document completeness (${pct(second.document_completeness,0)}) with the dataset average before changing automation policy.`)
 narrative.push(`Treat historical approval agreement as a process-consistency metric only. The source does not contain repayment/default outcomes, pricing or profitability.`)
 $('#segmentNarrative').innerHTML=narrative.map((x,i)=>`<div class="brief-item"><b>${i+1}</b><span>${esc(x)}</span></div>`).join('')
}

function setMode(mode){
 currentMode=mode; $('#enhancedMode').classList.toggle('active',mode==='enhanced');$('#legacyMode').classList.toggle('active',mode==='legacy');$('#enhancedForm').classList.toggle('hidden',mode!=='enhanced');$('#legacyForm').classList.toggle('hidden',mode!=='legacy');$('#setupBanner').classList.toggle('hidden',!(mode==='enhanced'&&!enhancedReady)); resetScoreResult()
}
$$('[data-mode]').forEach(b=>b.addEventListener('click',()=>setMode(b.dataset.mode)))
function resetScoreResult(){$('#scoreResult').className='panel result-panel empty';$('#scoreResult').innerHTML='<div class="empty-state"><div>◎</div><h3>Ready to score</h3><p>Prediction evidence and model explanation will appear here.</p></div>'}
function v1Payload(){const f=$('#legacyForm');return {married:f.married.value,dependents:f.dependents.value,education:f.education.value,self_employed:f.self_employed.value,applicant_income:Number(f.applicant_income.value),coapplicant_income:Number(f.coapplicant_income.value),loan_amount:Number(f.loan_amount.value),loan_amount_term:Number(f.loan_amount_term.value),credit_history:Number(f.credit_history.value),property_area:f.property_area.value}}
function v2Payload(){const f=$('#enhancedForm');return {no_of_dependents:Number(f.no_of_dependents.value),education:f.education.value,self_employed:f.self_employed.value,income_annum:Number(f.income_annum.value),loan_amount:Number(f.loan_amount.value),loan_term:Number(f.loan_term.value),cibil_score:Number(f.cibil_score.value),residential_assets_value:Number(f.residential_assets_value.value),commercial_assets_value:Number(f.commercial_assets_value.value),luxury_assets_value:Number(f.luxury_assets_value.value),bank_asset_value:Number(f.bank_asset_value.value)}}
function factorsHTML(factors=[]){const max=Math.max(...factors.map(x=>Math.abs(Number(x.contribution))),.001);return factors.map(x=>`<div class="factor-row"><div><span>${esc(x.label)}</span><span>${esc(x.direction)}</span></div><div class="factor-bar"><i class="${Number(x.contribution)<0?'negative':''}" style="width:${Math.max(5,Math.abs(Number(x.contribution))/max*100)}%"></i></div></div>`).join('')||'<p class="microcopy">No factor evidence returned.</p>'}
function renderScore(r, enhanced=false){const good=r.likely_approved;$('#scoreResult').className='panel result-panel';$('#scoreResult').innerHTML=`<div class="decision-head ${good?'good':'bad'}"><div><span class="eyebrow">${enhanced?'Enhanced ensemble':'Legacy baseline'}</span><h3>${good?'Likely approved':'Likely rejected'}</h3><p>${esc(r.champion_model)} · threshold ${pct(r.threshold,0)}</p></div><div class="prob-ring" style="--p:${Math.round(r.approval_probability*100)}%"><div><strong>${pct(r.approval_probability,0)}</strong><span>approval</span></div></div></div>${enhanced?`<div class="result-section"><h4>Ensemble members</h4>${(r.ensemble_members||[]).map(m=>`<p>${esc(m.model)} · ${pct(m.approval_probability)} · weight ${pct(m.weight,0)}</p>`).join('')}</div>`:''}<div class="result-section"><h4>Decision factors / local sensitivity</h4>${factorsHTML(r.top_factors)}</div><p class="microcopy">${esc(r.disclaimer)}</p>`}
$('#legacyForm').addEventListener('submit',async e=>{e.preventDefault();try{const r=await postJSON('/api/v1/predict',v1Payload());renderScore(r,false)}catch(err){$('#legacyError').textContent=err.message;$('#legacyError').classList.remove('hidden')}})
$('#enhancedForm').addEventListener('submit',async e=>{e.preventDefault();if(!enhancedReady){$('#enhancedError').textContent='Train the enhanced model first.';$('#enhancedError').classList.remove('hidden');return}try{const r=await postJSON('/api/v2/predict',v2Payload());renderScore(r,true)}catch(err){$('#enhancedError').textContent=err.message;$('#enhancedError').classList.remove('hidden')}})

function metricCard(label,value,note){return `<div class="metric-card"><span>${esc(label)}</span><strong>${esc(value)}</strong><small>${esc(note)}</small></div>`}
function renderModelEvidence(){
 if(!legacyInfo)return
 if(enhancedReady&&enhancedInfo?.metrics){
  const m=enhancedInfo.metrics.untouched_test||{};$('#modelMetrics').innerHTML=[metricCard('Final test accuracy',pct(m.accuracy),'Untouched 20% holdout'),metricCard('ROC-AUC',pct(m.roc_auc),'Ranking quality'),metricCard('Balanced accuracy',pct(m.balanced_accuracy),'Class-balanced view'),metricCard('Rejected recall',pct(m.recall_rejected),'Rejection sensitivity')].join('')
  $('#modelRows').innerHTML=(enhancedInfo.metrics.training_oof_candidates||[]).map(r=>`<tr><td>${esc(r.model)}</td><td>${pct(r.accuracy)}</td><td>${pct(r.roc_auc)}</td><td>${pct(r.balanced_accuracy)}</td></tr>`).join('')
  const imp=enhancedInfo.feature_importance||[], max=Math.max(...imp.map(x=>Math.max(0,Number(x.importance))),.001);$('#importanceList').innerHTML=imp.slice(0,10).map(x=>`<div class="importance-row"><header><span>${pretty(x.feature)}</span><strong>${Number(x.importance).toFixed(3)}</strong></header><div class="importance-track"><i style="width:${Math.max(2,Math.max(0,Number(x.importance))/max*100)}%"></i></div></div>`).join('');$('#modelProtocol').textContent=enhancedInfo.metadata.selection_protocol
 }else{
  const m=legacyInfo.metrics.out_of_fold_champion||{};$('#modelMetrics').innerHTML=[metricCard('OOF accuracy',pct(m.accuracy),'Legacy 5-fold OOF'),metricCard('ROC-AUC',pct(m.roc_auc),'Legacy baseline'),metricCard('Balanced accuracy',pct(m.balanced_accuracy),'Accounts for imbalance'),metricCard('Rejected recall',pct(m.recall_rejected),'Legacy rejection sensitivity')].join('')
  $('#modelRows').innerHTML=(legacyInfo.metrics.benchmark||[]).map(r=>`<tr><td>${esc(r.model)}</td><td>${pct(r.accuracy)}</td><td>${pct(r.roc_auc)}</td><td>${pct(r.balanced_accuracy)}</td></tr>`).join('')
  const imp=legacyInfo.feature_importance||[], max=Math.max(...imp.map(x=>Number(x.importance)),.001);$('#importanceList').innerHTML=imp.slice(0,10).map(x=>`<div class="importance-row"><header><span>${pretty(x.feature)}</span><strong>${pct(x.share)}</strong></header><div class="importance-track"><i style="width:${Math.max(2,Number(x.importance)/max*100)}%"></i></div></div>`).join('');$('#modelProtocol').textContent='Legacy evidence uses out-of-fold predictions and repeated stratified cross-validation. Train Enhanced v2 to unlock the separate 4,269-row locked-holdout benchmark.'
 }
}

async function boot(){
 updateControlLabels(); const state=$('#apiState')
 try{
  const h=await fetch('/api/health');if(!h.ok)throw new Error('API unavailable'); const health=await h.json(); state.className='api-state';state.querySelector('span').textContent='API online'
  const [v2,v1,ops]=await Promise.all([fetch('/api/v2/status').then(r=>r.json()),fetch('/api/v1/model').then(r=>r.json()),postJSON('/api/analysis/run_business_analysis',scenarioPayload())])
  enhancedReady=Boolean(v2.ready);enhancedInfo=v2;legacyInfo=v1;analysisData=ops;baselineAnalysis=ops;queueData=ops.queue||[]
  if(enhancedReady)setMode('enhanced');else setMode('legacy')
  renderCommand();renderPolicy();renderQueue();renderSegments();renderModelEvidence()
 }catch(err){console.error(err);state.className='api-state offline';state.querySelector('span').textContent='API offline';$('#executiveBrief').innerHTML=`<p class="error">${esc(err.message)}</p>`}
}
boot()
