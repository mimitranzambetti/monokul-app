// renderer.js

document.querySelector('#page-1').style.display = 'none'
document.querySelector('#page-2').style.display = 'none'
document.querySelector('#page-3').style.display = 'none'
document.querySelector('#page-error').style.display = 'none'

const zerorpc = require("zerorpc")
let client = new zerorpc.Client({heartbeatInterval: Math.pow(10, 6), timeout: Math.pow(10, 6)})
client.connect("tcp://127.0.0.1:4242")

let token = document.querySelector('#token')
let send_slack_token = document.querySelector('#send-slack-token')
let submit_custom_info = document.querySelector('#submit-custom-info')
let back = document.querySelector('#back')

let tokenValue = null
let slackChannelData = null

document.querySelector('#page-loading').style.display = 'none'
document.querySelector('#page-1').style.display = 'block'


send_slack_token.addEventListener('click', () => {
  tokenValue = token.value
  client.invoke("send_slack_token", token.value, (error, res) => {
    if(error) {
      console.error(error)
    } else {
      data = JSON.parse(res)
      console.log(data)

      for (i = 0; i < data.channels.length; i++) {
        document.querySelector('#channels').innerHTML = document.querySelector('#channels').innerHTML.concat(
          '<tr class="">\
            <td>#' + data.channels[i].name + '</td>\
            <td>\
              <input type="radio" id="yes" name=channel_"' + data.channels[i].name + '" value="yes" checked="checked">\
              <label for="channel_"' + data.channels[i].name + '">Yes   </label>\
              <input type="radio" id="no" name=channel_"' + data.channels[i].name + '" value="no">\
              <label for="channel_"' + data.channels[i].name + '">No   </label>\
            </td>\
          </tr>\n'
        )
      }
      for (const k in data.users) {
        document.querySelector('#gender-tags').innerHTML = document.querySelector('#gender-tags').innerHTML.concat(
          '<tr>\
            <td>' + data.users[k] + '</td>\
            <td>\
              <select name="gender">\
                <option value="woman">Woman</option>\
                <option value="man">Man</option>\
                <option value="other">Other</option>\
                <option value="Prefer not to say" selected="selected">Prefer not to say</option>\
              </select>\
            </td>\
          </tr>'
        )
      }
      document.querySelector('#page-loading').style.display = 'none'
      document.querySelector('#page-2').style.display = 'block'
    }
  })
  document.querySelector('#page-1').style.display = 'none'
  document.querySelector('#page-loading').style.display = 'block'
})

submit_custom_info.addEventListener('click', () => {
  client.invoke("run_analyses", token.value, (error, res) => {
    if(error) {
      console.error(error)
    } else {
      console.log(String(res))
      document.querySelector('#page-loading').style.display = 'none'
      document.querySelector('#page-3').style.display = 'block'
    }
  })
  document.querySelector('#page-2').style.display = 'none'
  document.querySelector('#page-loading').style.display = 'block'
})

back.addEventListener('click', () => {
  document.querySelector('#page-2').style.display = 'none'
  document.querySelector('#page-1').style.display = 'block'
})