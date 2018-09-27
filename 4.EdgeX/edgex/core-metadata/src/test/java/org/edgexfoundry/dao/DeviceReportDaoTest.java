/*******************************************************************************
 * Copyright 2016-2017 Dell Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 * @microservice: support-logging
 * @author: Jim White, Dell
 * @version: 1.0.0
 *******************************************************************************/

package org.edgexfoundry.dao;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.mockito.Mockito.when;

import org.edgexfoundry.domain.meta.Device;
import org.edgexfoundry.domain.meta.DeviceReport;
import org.edgexfoundry.domain.meta.DeviceService;
import org.edgexfoundry.test.category.RequiresNone;
import org.edgexfoundry.test.data.DeviceData;
import org.edgexfoundry.test.data.ReportData;
import org.edgexfoundry.test.data.ServiceData;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

@Category(RequiresNone.class)
public class DeviceReportDaoTest {

  private static final String TEST_ID = "123";

  @InjectMocks
  private DeviceReportDao dao;

  @Mock
  private DeviceReportRepository repos;

  @Mock
  private DeviceRepository deviceRepos;

  private DeviceReport report;

  @Before
  public void setup() throws Exception {
    MockitoAnnotations.initMocks(this);
    report = ReportData.newTestInstance();
  }

  @Test
  public void testGetByIdOrNameWithNull() {
    assertNull("Returned device report is not null", dao.getByIdOrName(null));
  }

  @Test
  public void testGetByIdOrName() {
    when(repos.findOne(TEST_ID)).thenReturn(report);
    report.setId(TEST_ID);
    assertEquals("Returned device report is not expected", report, dao.getByIdOrName(report));
  }

  @Test
  public void testGetByIdOrNameWithNoId() {
    when(repos.findByName(ReportData.TEST_RPT_NAME)).thenReturn(report);
    assertEquals("Returned device report is not expected", report, dao.getByIdOrName(report));
  }

  @Test
  public void testGetOwningService() {
    DeviceService service = ServiceData.newTestInstance();
    Device device = DeviceData.newTestInstance();
    device.setService(service);
    report.setDevice(DeviceData.TEST_NAME);
    when(deviceRepos.findByName(DeviceData.TEST_NAME)).thenReturn(device);
    assertEquals("Service returned not as expected", service, dao.getOwningService(report));
  }

  @Test
  public void testGetOwningServiceWithNull() {
    assertNull("Service returned should have been null", dao.getOwningService(null));
  }

  @Test
  public void testRemoveAssociatedReportsForDevice() {
    Device device = DeviceData.newTestInstance();
    dao.removeAssociatedReportsForDevice(device);
  }

  @Test
  public void testGetValueDescriptorsForDeviceReportsAssociatedToDevice() {
    dao.getValueDescriptorsForDeviceReportsAssociatedToDevice(DeviceData.TEST_NAME);
  }
}
