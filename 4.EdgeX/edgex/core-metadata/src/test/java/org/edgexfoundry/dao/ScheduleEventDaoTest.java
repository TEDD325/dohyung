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
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.mockito.Mockito.when;

import java.util.ArrayList;
import java.util.List;

import org.edgexfoundry.domain.meta.DeviceReport;
import org.edgexfoundry.domain.meta.DeviceService;
import org.edgexfoundry.domain.meta.ScheduleEvent;
import org.edgexfoundry.test.category.RequiresNone;
import org.edgexfoundry.test.data.ReportData;
import org.edgexfoundry.test.data.ScheduleEventData;
import org.edgexfoundry.test.data.ServiceData;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

@Category(RequiresNone.class)
public class ScheduleEventDaoTest {

  private static final String TEST_ID = "123";

  @InjectMocks
  private ScheduleEventDao dao;

  @Mock
  private ScheduleEventRepository repos;

  @Mock
  private DeviceReportRepository reportRepos;

  @Mock
  private DeviceServiceRepository deviceServiceRepos;

  private ScheduleEvent event;

  @Before
  public void setup() throws Exception {
    MockitoAnnotations.initMocks(this);
    event = ScheduleEventData.newTestInstance();
  }

  @Test
  public void testGetByIdOrNameWithNull() {
    assertNull("Returned event is not null", dao.getByIdOrName(null));
  }

  @Test
  public void testGetByIdOrName() {
    when(repos.findOne(TEST_ID)).thenReturn(event);
    event.setId(TEST_ID);
    assertEquals("Returned event is not expected", event, dao.getByIdOrName(event));
  }

  @Test
  public void testGetByIdOrNameWithNoId() {
    when(repos.findByName(ScheduleEventData.TEST_SCHEDULE_EVENT_NAME)).thenReturn(event);
    assertEquals("Returned event is not expected", event, dao.getByIdOrName(event));
  }

  @Test
  public void testIsScheduleEventAssociatedToDeviceReport() {
    List<DeviceReport> reports = new ArrayList<>();
    DeviceReport report = ReportData.newTestInstance();
    reports.add(report);
    when(reportRepos.findByEvent(ScheduleEventData.TEST_SCHEDULE_EVENT_NAME)).thenReturn(reports);
    assertTrue("Association to device report did not report true",
        dao.isScheduleEventAssociatedToDeviceReport(event));
  }

  @Test
  public void testIsScheduleEventAssociatedToDeviceReportWithNull() {
    assertFalse("Association to device report should not be true with null",
        dao.isScheduleEventAssociatedToDeviceReport(null));
  }

  @Test
  public void testGetAffectedService() {
    List<DeviceService> services = new ArrayList<>();
    DeviceService service = ServiceData.newTestInstance();
    event.setService(ServiceData.TEST_SERVICE_NAME);
    services.add(service);
    when(deviceServiceRepos.findByName(ServiceData.TEST_SERVICE_NAME)).thenReturn(service);
    assertEquals("Returned device services do not match expected return list", services,
        dao.getAffectedService(event));
  }

  @Test
  public void testGetAffectedServiceWithServiceNotFound() {
    List<DeviceService> services = new ArrayList<>();
    event.setService(ServiceData.TEST_SERVICE_NAME);
    when(deviceServiceRepos.findByName(ServiceData.TEST_SERVICE_NAME)).thenReturn(null);
    assertEquals("Returned device services do not match expected return list", services,
        dao.getAffectedService(event));
  }

  @Test
  public void testGetAffectedServiceWithNull() {
    List<DeviceService> services = new ArrayList<>();
    assertEquals("Returned device services do not match expected return list", services,
        dao.getAffectedService(null));
  }

}
