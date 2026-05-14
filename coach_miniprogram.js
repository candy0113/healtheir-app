// pages/coach/coach.js
// HEalthieR 语音教练主页面 — 微信小程序

const API_BASE = 'https://api.yourdomain.com/v1'

const SCENE_MAP = {
  gym:     { label: '力量训练', topic: '运动健身',   bgColor: '#FBEAF0' },
  kitchen: { label: '营养厨房', topic: '营养策略',   bgColor: '#E1F5EE' },
  rest:    { label: '补剂时间', topic: '补剂与认知', bgColor: '#EEEDFE' },
}

Page({
  data: {
    // 场景
    currentScene: 'gym',
    scenes: Object.entries(SCENE_MAP).map(([k, v]) => ({ key: k, ...v })),

    // 角色状态: idle | listening | thinking | speaking
    avatarState: 'idle',
    avatarAction: 'idle_breathe',

    // 对话
    question: '',
    answer: '',
    isThinking: false,

    // 音频
    audioSrc: '',

    // UI
    micPressed: false,
    subtitleText: '选择场景，按住麦克风提问',
  },

  onLoad() {
    this.recorderManager = wx.getRecorderManager()
    this.innerAudioContext = wx.createInnerAudioContext()
    this._setupRecorder()
    this._setupAudio()
  },

  onUnload() {
    this.innerAudioContext.destroy()
  },

  // ─── 场景切换 ─────────────────────────────────────────────────────────────

  switchScene(e) {
    const scene = e.currentTarget.dataset.scene
    this.setData({ currentScene: scene })
    wx.vibrateShort({ type: 'light' })
  },

  // ─── 录音设置 ─────────────────────────────────────────────────────────────

  _setupRecorder() {
    this.recorderManager.onStart(() => {
      this.setData({ avatarState: 'listening', micPressed: true, subtitleText: '正在聆听...' })
    })

    this.recorderManager.onStop(async (res) => {
      this.setData({ micPressed: false, avatarState: 'thinking', subtitleText: '思考中...' })
      await this._processAudio(res.tempFilePath)
    })

    this.recorderManager.onError((err) => {
      console.error('[Recorder]', err)
      wx.showToast({ title: '录音出错，请重试', icon: 'none' })
      this.setData({ avatarState: 'idle', subtitleText: '按住麦克风提问' })
    })
  },

  _setupAudio() {
    this.innerAudioContext.onPlay(() => {
      this.setData({ avatarState: 'speaking' })
    })
    this.innerAudioContext.onEnded(() => {
      this.setData({ avatarState: 'idle', subtitleText: '按住麦克风继续提问' })
    })
    this.innerAudioContext.onError((err) => {
      console.error('[Audio]', err)
      this.setData({ avatarState: 'idle' })
    })
  },

  // ─── 麦克风按住/松开 ────────────────────────────────────────────────────────

  onMicTouchStart() {
    wx.authorize({ scope: 'scope.record' }).then(() => {
      this.recorderManager.start({
        duration: 30000,       // 最长30秒
        sampleRate: 16000,
        numberOfChannels: 1,
        encodeBitRate: 48000,
        format: 'mp3',
      })
    }).catch(() => {
      wx.showModal({
        title: '需要麦克风权限',
        content: '请在设置中开启麦克风权限',
        showCancel: false,
      })
    })
  },

  onMicTouchEnd() {
    this.recorderManager.stop()
  },

  // ─── 音频全链路处理 ──────────────────────────────────────────────────────────

  async _processAudio(tempFilePath) {
    const scene = SCENE_MAP[this.data.currentScene]
    try {
      // 上传录音到后端全链路接口
      const res = await this._uploadAudio(tempFilePath, scene.topic)

      this.setData({
        question: res.question,
        answer: res.answer,
        subtitleText: res.answer,
        avatarAction: res.avatar_action?.animation || 'speak',
      })

      // 播放 TTS 音频（base64 → 临时文件）
      if (res.audio_base64) {
        await this._playBase64Audio(res.audio_base64)
      }
    } catch (err) {
      console.error('[Pipeline]', err)
      wx.showToast({ title: '网络出错，请重试', icon: 'none' })
      this.setData({ avatarState: 'idle', subtitleText: '按住麦克风提问' })
    }
  },

  _uploadAudio(filePath, topic) {
    return new Promise((resolve, reject) => {
      const token = wx.getStorageSync('auth_token')
      wx.uploadFile({
        url: `${API_BASE}/voice/full-pipeline?scene=${encodeURIComponent(topic)}`,
        filePath,
        name: 'file',
        header: {
          Authorization: `Bearer ${token}`,
        },
        success: (res) => {
          try {
            resolve(JSON.parse(res.data).data)
          } catch (e) {
            reject(e)
          }
        },
        fail: reject,
      })
    })
  },

  async _playBase64Audio(base64Str) {
    // 将 base64 写入临时文件
    const fs = wx.getFileSystemManager()
    const tmpPath = `${wx.env.USER_DATA_PATH}/coach_reply_${Date.now()}.mp3`
    await new Promise((resolve, reject) => {
      fs.writeFile({
        filePath: tmpPath,
        data: base64Str,
        encoding: 'base64',
        success: resolve,
        fail: reject,
      })
    })
    this.innerAudioContext.src = tmpPath
    this.innerAudioContext.play()
  },

  // ─── 快速话题点击（不需要语音）────────────────────────────────────────────

  async onTopicTap(e) {
    const text = e.currentTarget.dataset.text
    this.setData({ avatarState: 'thinking', subtitleText: '思考中...', question: text })

    const scene = SCENE_MAP[this.data.currentScene]
    const token = wx.getStorageSync('auth_token')

    try {
      // 文字 RAG 查询
      const ragRes = await this._request('/rag/query', 'POST', { text, scene: scene.topic }, token)
      const answer = ragRes.answer

      // TTS 转语音
      const ttsRes = await this._request('/tts', 'POST', { text: answer, voice: 'shimmer' }, token)

      this.setData({ answer, subtitleText: answer })
      if (ttsRes.audio_base64) {
        await this._playBase64Audio(ttsRes.audio_base64)
      }
    } catch (err) {
      wx.showToast({ title: '出错了，请重试', icon: 'none' })
      this.setData({ avatarState: 'idle', subtitleText: '按住麦克风提问' })
    }
  },

  _request(path, method, data, token) {
    return new Promise((resolve, reject) => {
      wx.request({
        url: `${API_BASE}${path}`,
        method,
        data,
        header: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
        success: (res) => resolve(res.data.data),
        fail: reject,
      })
    })
  },
})
