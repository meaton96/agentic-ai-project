import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { McpPage } from './McpPage'
import * as client from '../api/client'
import type { McpConfigResponse } from '../api/types'

vi.mock('../api/client')

const CONFIG: McpConfigResponse = {
  config_path: '/repo/agentic-ml-classification/configs/mcp_server.json',
  config_exists: true,
  name: 'agentic-ml-facts',
  host: '127.0.0.1',
  port: 8765,
  url: 'http://127.0.0.1:8765/mcp',
  enabled_tools: ['get_raw_schema', 'list_templates'],
  reachable: false,
  tools: [
    {
      name: 'get_raw_schema',
      description: "Get the dataset's raw column facts for this run.",
      input_schema: { type: 'object', properties: { run_id: { type: 'string' } }, required: ['run_id'] },
      enabled: true,
    },
    {
      name: 'get_flight_deep_dive_evidence',
      description: 'Get the full deep-dive evidence for one flagged flight in this run.',
      input_schema: { type: 'object', properties: {} },
      enabled: false,
    },
  ],
}

describe('McpPage', () => {
  it('renders server config and a card per tool with enabled/disabled state', async () => {
    vi.mocked(client.getMcpConfig).mockResolvedValue(CONFIG)
    render(<McpPage />)

    expect(await screen.findByRole('heading', { name: 'agentic-ml-facts' })).toBeInTheDocument()
    expect(screen.getByText('not reachable')).toBeInTheDocument()
    expect(screen.getByText('http://127.0.0.1:8765/mcp')).toBeInTheDocument()

    const enabledCard = screen.getByText('get_raw_schema').closest('.card') as HTMLElement
    expect(within(enabledCard).getByText('enabled')).toBeInTheDocument()

    const disabledCard = screen.getByText('get_flight_deep_dive_evidence').closest('.card') as HTMLElement
    expect(within(disabledCard).getByText('disabled')).toBeInTheDocument()
  })

  it('expands a tool card to show its input schema', async () => {
    vi.mocked(client.getMcpConfig).mockResolvedValue(CONFIG)
    const user = userEvent.setup()
    render(<McpPage />)

    const card = (await screen.findByText('get_raw_schema')).closest('.card') as HTMLElement
    expect(within(card).queryByText(/"run_id"/)).not.toBeInTheDocument()

    await user.click(within(card).getByText('Input schema'))
    expect(within(card).getByText(/"run_id"/)).toBeInTheDocument()
  })

  it('shows an error message if the config request fails', async () => {
    vi.mocked(client.getMcpConfig).mockRejectedValue(new Error('network down'))
    render(<McpPage />)
    expect(await screen.findByText(/Failed to load MCP config: network down/)).toBeInTheDocument()
  })
})
