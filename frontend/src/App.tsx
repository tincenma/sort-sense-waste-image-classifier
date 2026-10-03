import { useEffect, useRef, useState } from 'react'
import { ArrowDown, ArrowRight, Check, CircleCheck, ImagePlus, Leaf, LoaderCircle, RefreshCw, Upload, X } from 'lucide-react'
import { API_BASE, classify, fetchExample, fetchInfo } from './api'
import type { AppInfo, Category, Example, ModelId, PredictionResponse } from './types'

const labels: Record<Category, string> = {
  cardboard: 'Cardboard', glass: 'Glass', metal: 'Metal', paper: 'Paper', plastic: 'Plastic', trash: 'Trash',
}
const descriptions: Record<Category, string> = {
  cardboard: 'Boxes, packaging and corrugated board.',
  glass: 'Glass bottles, jars and containers.',
  metal: 'Cans and other metal objects.',
  paper: 'Sheets, newspapers and paper products.',
  plastic: 'Plastic bottles, bags and containers.',
  trash: 'Other waste in the dataset.',
}
const percent = (score: number) => `${(score * 100).toFixed(1)}%`
const connectionMessage = 'The classifier is unavailable. Check that the backend is running, then reconnect.'

function errorMessage(error: unknown): string {
  if (error instanceof TypeError) return connectionMessage
  return error instanceof Error ? error.message : 'Something went wrong. Please try again.'
}

export default function App() {
  const [info, setInfo] = useState<AppInfo | null>(null)
  const [connectionError, setConnectionError] = useState('')
  const [connectionAttempt, setConnectionAttempt] = useState(0)
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState('')
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [selectedModel, setSelectedModel] = useState<ModelId>('mobilenetv2')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const activeRequest = useRef<AbortController | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    setConnectionError('')
    fetchInfo(controller.signal).then(setInfo).catch((err: unknown) => {
      if (!controller.signal.aborted) setConnectionError(errorMessage(err))
    })
    return () => controller.abort()
  }, [connectionAttempt])

  useEffect(() => {
    if (!file) { setPreview(''); return }
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  useEffect(() => () => activeRequest.current?.abort(), [])

  function cancelRequest() {
    activeRequest.current?.abort()
    activeRequest.current = null
    setBusy(false)
  }

  function chooseFile(nextFile: File): boolean {
    cancelRequest()
    setError('')
    setResult(null)
    const allowedType = ['image/jpeg', 'image/png', 'image/webp'].includes(nextFile.type)
    const allowedExtension = /\.(jpe?g|png|webp)$/i.test(nextFile.name)
    if (!allowedType && !(nextFile.type === '' && allowedExtension)) {
      setError('Please choose a JPG, PNG, or WebP image.')
      setFile(null)
      return false
    }
    if (!nextFile.size || nextFile.size > (info?.max_file_mb ?? 10) * 1024 * 1024) {
      setError(nextFile.size ? 'This image is too large. The limit is 10 MB.' : 'This image is empty. Please choose another.')
      setFile(null)
      return false
    }
    setFile(nextFile)
    return true
  }

  async function analyse(nextFile: File) {
    cancelRequest()
    const controller = new AbortController()
    activeRequest.current = controller
    setBusy(true)
    setError('')
    setResult(null)
    try {
      const prediction = await classify(nextFile, controller.signal)
      if (!controller.signal.aborted) setResult(prediction)
    } catch (err) {
      if (!controller.signal.aborted) setError(errorMessage(err))
    } finally {
      if (activeRequest.current === controller) {
        setBusy(false)
        activeRequest.current = null
      }
    }
  }

  async function tryExample(example: Example) {
    cancelRequest()
    const controller = new AbortController()
    activeRequest.current = controller
    setBusy(true)
    setError('')
    setResult(null)
    try {
      const blob = await fetchExample(example.image_url, controller.signal)
      if (controller.signal.aborted) return
      const exampleFile = new File([blob], `${example.id}.jpg`, { type: 'image/jpeg' })
      setFile(exampleFile)
      const prediction = await classify(exampleFile, controller.signal)
      if (!controller.signal.aborted) setResult(prediction)
    } catch (err) {
      if (!controller.signal.aborted) setError(errorMessage(err))
    } finally {
      if (activeRequest.current === controller) {
        setBusy(false)
        activeRequest.current = null
      }
    }
  }

  function reset() {
    cancelRequest()
    setFile(null)
    setResult(null)
    setError('')
    if (inputRef.current) inputRef.current.value = ''
  }

  const prediction = result?.predictions.find(item => item.model_id === selectedModel)
  const scores = prediction ? Object.entries(prediction.category_scores).sort((a, b) => b[1] - a[1]) as [Category, number][] : []
  const agreement = result?.predictions.length === 2 && result.predictions[0].predicted_category === result.predictions[1].predicted_category

  return (
    <>
      <header className="site-header">
        <a className="brand" href="#" aria-label="SortSense home"><span className="brand-icon"><Leaf size={23} /></span>SortSense<span className="brand-dot">.</span></a>
        <nav aria-label="Main navigation"><a href="#try-it">Try it</a><a href="#the-models">The models</a></nav>
        <div className={`connection ${info ? 'connected' : connectionError ? 'offline' : ''}`}><span />{info ? 'Ready to classify' : connectionError ? 'Backend offline' : 'Connecting…'}</div>
      </header>

      <main>
        <section className="hero" aria-labelledby="hero-title">
          <div className="eyebrow"><span className="tiny-leaf"><Leaf size={13} /></span> COMPUTER VISION FOR WASTE SORTING</div>
          <h1 id="hero-title">One image.<br /><span>Two perspectives.</span></h1>
          <div className="hero-bottom"><p>What does your waste look like to AI? Upload a photo and see how two trained models sort it into six categories.</p><a className="jump-link" href="#try-it">Let’s find out <ArrowDown size={18} /></a></div>
        </section>

        {connectionError && <div className="connection-alert" role="alert"><p>{connectionError}</p><button className="text-button" onClick={() => setConnectionAttempt(value => value + 1)}><RefreshCw size={15} /> Reconnect</button></div>}

        <section className="workspace" id="try-it" aria-label="Image classifier">
          <div className="upload-card card">
            <div className="card-heading"><div><span className="step-number">01</span><h2>Your image</h2></div><span className="small-label">JPG · PNG · WEBP</span></div>
            <input ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp" className="visually-hidden" aria-label="Choose a waste image" onChange={event => { const next = event.target.files?.[0]; if (next) chooseFile(next); event.target.value = '' }} />
            <div className={`upload-zone ${preview ? 'has-image' : ''} ${dragging ? 'dragging' : ''}`} onDragOver={event => { event.preventDefault(); setDragging(true) }} onDragLeave={event => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDragging(false) }} onDrop={event => { event.preventDefault(); setDragging(false); const next = event.dataTransfer.files[0]; if (next) chooseFile(next) }}>
              {preview ? <><img className="image-preview" src={preview} alt={`Selected image: ${file?.name}`} /><button className="remove-image" onClick={reset} aria-label="Remove image"><X size={17} /></button><button className="change-image" onClick={() => inputRef.current?.click()}><ImagePlus size={15} /> Change image</button></> : <div className="upload-placeholder"><span className="upload-icon"><Upload size={29} strokeWidth={1.5} /></span><h3>Drop an image here</h3><p>A clear photo of one item works best.</p><button className="outline-button" onClick={() => inputRef.current?.click()}>Choose a photo <ImagePlus size={16} /></button><span className="file-limit">Up to 10 MB · still images</span></div>}
            </div>
            <div className="file-details"><span>{file ? file.name : 'Your next sorting decision starts here.'}</span>{file && <span>{result ? `${result.width} × ${result.height} · ` : ''}{file.size < 1024 * 1024 ? `${Math.round(file.size / 1024)} KB` : `${(file.size / 1024 / 1024).toFixed(1)} MB`}</span>}</div>
            {error && <p className="error-message" role="alert">{error}</p>}
            <button className="primary-button" disabled={!file || busy || !info} onClick={() => file && void analyse(file)}>{busy ? <><LoaderCircle size={18} className="spinner" /> Running both models…</> : <>Analyse image <ArrowRight size={18} /></>}</button>
            <p className="privacy-note">Your image is processed for this prediction and is not saved.</p>
          </div>

          <div className="prediction-card card" aria-busy={busy}>
            <div className="card-heading"><div><span className="step-number">02</span><h2>See the prediction</h2></div><span className="small-label">6 CATEGORIES</span></div>
            <div className="model-tabs" role="tablist" aria-label="Prediction model">{(['mobilenetv2', 'baseline_cnn'] as ModelId[]).map(id => <button key={id} role="tab" id={`tab-${id}`} aria-selected={selectedModel === id} aria-controls="prediction-panel" className={selectedModel === id ? 'active' : ''} onClick={() => setSelectedModel(id)}>{id === 'mobilenetv2' ? 'MobileNetV2' : 'Small CNN'}{id === 'mobilenetv2' && <span>TOP MODEL</span>}</button>)}</div>
            <div id="prediction-panel" role="tabpanel" aria-labelledby={`tab-${selectedModel}`} aria-live="polite">
              {busy ? <div className="result-placeholder"><div className="placeholder-symbol"><LoaderCircle size={32} className="spinner" /></div><h3>Looking for familiar patterns</h3><p>Both models are analysing your image.<br />The six category scores will appear here.</p></div> : prediction ? <div className="result-content">
                <div className="prediction-summary"><div><span className="result-kicker">PREDICTED CATEGORY</span><h3>{labels[prediction.predicted_category]}</h3><p>{descriptions[prediction.predicted_category]}</p></div><div className="top-score"><strong>{percent(prediction.model_score)}</strong><span>model score</span></div></div>
                <div className="score-heading"><span>Category scores</span><span>Higher = stronger match</span></div>
                <div className="score-chart" aria-label="Scores for all six categories">{scores.map(([category, score], index) => <div className={`score-row ${index === 0 ? 'highest' : ''}`} key={category}><div className="score-label"><span>{labels[category]}{index === 0 && <Check size={13} />}</span><span>{percent(score)}</span></div><div className="bar-track"><div className="bar-fill" style={{ width: `${score * 100}%` }} /></div></div>)}</div>
                <p className="score-note">Scores show each model’s preference, not a guarantee that its answer is correct.</p>
              </div> : <div className="result-placeholder"><div className="placeholder-symbol"><Leaf size={33} strokeWidth={1.5} /></div><h3>A little perspective on your waste</h3><p>Choose a photo or try an example below.<br />We’ll compare the models side by side.</p><div className="category-pills">{Object.values(labels).map(label => <span key={label}>{label}</span>)}</div></div>}
            </div>
          </div>
        </section>

        {result && <section className="comparison" aria-label="Model comparison"><div className="comparison-title"><CircleCheck size={20} /><div><h3>{agreement ? 'Both models agree' : 'Two models, different answers'}</h3><p>{agreement ? 'They selected the same category for this image.' : 'Compare the scores and check the item yourself.'}</p></div></div><div className="comparison-predictions">{result.predictions.map(item => <div key={item.model_id}><span>{item.model_id === 'mobilenetv2' ? 'MobileNetV2' : 'Small CNN'}</span><strong>{labels[item.predicted_category]} <small>{percent(item.model_score)}</small></strong></div>)}</div><span className="timing">{(result.total_ms / 1000).toFixed(2)}s inference</span></section>}

        <section className="examples" aria-labelledby="examples-title"><div className="section-heading"><h2 id="examples-title">No photo handy? Try an example.</h2><span>Real images from our TrashNet test set</span></div><div className="example-grid">{info?.examples.map(example => <button className="example-card" key={example.id} onClick={() => void tryExample(example)} disabled={busy}><div><img src={`${API_BASE}${example.image_url}`} alt={`${labels[example.label]} example`} loading="lazy" /><span className="example-arrow"><ArrowRight size={15} /></span></div><span>{labels[example.label]}</span></button>) ?? <p className="examples-wait">Examples will appear when the backend connects.</p>}</div></section>

        <section className="how-it-works" aria-labelledby="how-title"><div className="section-intro"><span className="eyebrow">FROM PIXELS TO A PREDICTION</span><h2 id="how-title">A quick look inside.</h2></div><div className="how-grid"><article><span>01 / PREPARE</span><h3>Start with the image</h3><p>The photo is converted to RGB and resized to 224 × 224 pixels, just like the training images.</p></article><article><span>02 / RECOGNISE</span><h3>Look for patterns</h3><p>Each model uses the shapes, colours and textures it learned to recognise from labelled waste images.</p></article><article><span>03 / COMPARE</span><h3>Score all six categories</h3><p>The category with the highest score becomes the prediction. Switch tabs to explore each model’s scores.</p></article></div></section>

        <section className="models-section" id="the-models" aria-labelledby="models-title"><div className="section-heading"><div><span className="eyebrow">TRAINED. TESTED. COMPARED.</span><h2 id="models-title">Meet the models.</h2></div><p>Measured on 378 held-out TrashNet images.<br />These are test results, not scores for your photo.</p></div><div className="models-grid">{info?.models.map(model => <article className={`model-card ${model.id === 'mobilenetv2' ? 'featured' : ''}`} key={model.id}><div className="model-title"><h3>{model.name}</h3><span>{model.id === 'mobilenetv2' ? 'TRANSFER LEARNING' : 'BASELINE'}</span></div><p>{model.description}</p><div className="model-metrics"><div><strong>{percent(model.test_accuracy)}</strong><span>Test accuracy</span></div><div><strong>{model.test_macro_f1.toFixed(3)}</strong><span>Macro F1</span></div></div><div className="model-footnote">{model.id === 'mobilenetv2' ? <><Check size={15} /> Selected using validation macro F1</> : 'Trained from scratch on the same image split'}</div></article>) ?? <p>Connect the backend to see measured model results.</p>}</div></section>
      </main>
      <footer><a className="brand" href="#"><Leaf size={20} />SortSense<span className="brand-dot">.</span></a><p>Six categories. Two models. A closer look at waste.</p><span>Computer vision project</span></footer>
    </>
  )
}
